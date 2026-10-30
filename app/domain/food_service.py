import os, logging, cv2, shutil
import traceback, string, json
import numpy as np
import app.domain.common.image_utils as ImageUtils
import app.domain.common.timer_utils as utils
import app.domain.common.gen_foodtree as foodTree
from app.domain.pipeline.a_onnx_segment import OnnxSegmentPy
from app.domain.pipeline.b_get_food_area import FoodRoiExactor
from app.domain.pipeline.c_onnx_classify import OnnxClassifier
from app.domain.pipeline.d_clipdtct import ClipDetect
from app.domain.pipeline.e_embedding import OnnxEmbedding
from app.domain.pipeline.f_machine import OnnxMachine
import app.domain.common.timer_utils as timerUtils
from datetime import datetime
from app.core.config.settings import settings
from app.core.observability.log_helper import show_text
from PIL import Image
import time, uuid


logger = logging.getLogger("food_service")

def is_uuid(value: str) -> bool:
    """
    判断字符串是否为合法 UUID
    """
    if not isinstance(value, str):
        return False
    try:
        uuid_obj = uuid.UUID(value)
        return str(uuid_obj) == value.lower()
    except (ValueError, AttributeError, TypeError):
        return False
    

class FoodService:
    def __init__(self, cfg):
        self.segmentor = OnnxSegmentPy(cfg)
        self.roi_extractor = FoodRoiExactor(cfg)
        self.shengShu = OnnxClassifier(cfg)
        self.clip = ClipDetect(cfg)
        self.embedding = OnnxEmbedding(cfg)
        self.machine = OnnxMachine(cfg)
        self.iniDetectRange()
        self.machine_range = cfg['TOTAL_CONFIG']['machine_list']
        self.food_tree = foodTree.IngredientTreeBuilder("config/param/food_tree.txt", self.src2tgt)

    def iniDetectRange(self):
        self.get_change_names()

        range_path = './config/model/clip/labels.txt'
        self.detect_range = {}
        keys = self.src2tgt.values()
        with open(range_path, "r", encoding="utf-8") as f:
            for line in f.readlines():
                line = line.strip()
                has_change = False
                for key in keys:
                    if (key in line) or (key == line):
                        has_change = True
                        tgt_name = line.replace(key, self.tgt2src[key])
                        self.detect_range[line] = tgt_name
                        logger.info('添加detect_range: %s --> %s', line, tgt_name)
                        break
                
                if 'N' in line:
                    has_change = True
                    new_name = line.replace('N', '')
                    self.detect_range[line] = new_name
                    logger.info('添加detect_range: %s --> %s', line, new_name)
                                
                if not has_change:
                    self.detect_range[line] = line

    def get_change_names(self):
        file_path = './config/model/clip/changeVocab.txt'
        self.src2tgt = {}
        self.tgt2src = {}
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                if line.startswith("#"):
                    continue
                items = line.split()
                self.src2tgt[items[0]] = items[1]
                self.tgt2src[items[1]] = items[0]
        logger.info('change_names: %s', self.src2tgt)

    def single_detect_tree(self, image_datas, save_result: bool = False):
        img = image_datas['crop']
        detect_range = list(self.detect_range.keys())
        clip_res = self.clip.detect_with_text_model(img, detect_range)
        final_res = []
        total_score = 0.0
        use_images = []
        for text, score in clip_res:
            mapped_name = self.detect_range.get(text, text)
            final_res.append((mapped_name, score))
            if mapped_name in ['空腔', '无食材']:
                total_score += score
            else:
                use_images.append((mapped_name, score))
        if len(final_res) == 0 or len(use_images) == 0:
            use_images = [('空腔', 1.0)]
        elif total_score >= 0.1:
            use_images = [('空腔', total_score)]
            
        result = " ".join([f"{name}: {score:.3f}" for name, score in clip_res])
        if save_result:
            crop_img = image_datas['visual_res']
            crop_img = ImageUtils.draw_cn_text(crop_img, result, (10, 10), font_size=20)
            image_datas['visual_res'] = crop_img 
        image_datas['clip_res'] = use_images
        return image_datas

    def _save_detection_result(self, clip_res, bgr_ori):
        save_root = settings.paths.save_images_dir
        logger.info("图像保存路径: %s", save_root)
        os.makedirs(save_root, exist_ok=True)
        save_datas = {
            'vis': clip_res["visual_res"],
            'lbl': clip_res["mask"],
            'crop': clip_res['crop'],
        }
        
        msk_path = os.path.join(save_root, "01-msk.jpg")
        cv2.imwrite(msk_path, clip_res["mask"] * 125)
        vis_path = os.path.join(save_root, "00-vis.jpg")
        cv2.imwrite(vis_path, clip_res["visual_res"])
        crop_path = os.path.join(save_root, "02-crop.jpg")
        cv2.imwrite(crop_path, clip_res['crop'])
        now_time = datetime.now().strftime("%y%m%d")
        save_folder = os.path.join(save_root, now_time)
        os.makedirs(save_folder, exist_ok=True)
        base_name = datetime.now().strftime("%y%m%d%H%M%S")
        filename = f"turbo_{base_name}.jpg"
        save_path = os.path.join(save_folder, filename)
        if os.path.exists(save_path):
            for letter in string.ascii_lowercase:
                filename = f"{base_name}{letter}.jpg"
                save_path = os.path.join(save_folder, filename)
                if not os.path.exists(save_path):
                    break
        cv2.imwrite(save_path, bgr_ori)
        return save_datas
    
    
    def checkShengShu(self, image_data: bytes, machine_info: str = "", trace_id: str = "", save_result: bool = False):
        logger.info('start, trace_id: %s, save_result: %s, save_env: %s, machine_info: %s', trace_id, save_result, settings.paths.save_images_enabled, machine_info)
        save_result = save_result or settings.paths.save_images_enabled
        timer_tools = timerUtils.timer_tool()
        img_bgr = None
        if isinstance(image_data, bytes):
            buf = np.frombuffer(image_data, dtype=np.uint8)
            img_bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        res, val = self.shengShu.isShu(img_bgr)
        logger.info("isShu: %d", res)
        timer_tools.mark('shengShu')
        timer_summary = timer_tools.summary()
        show_text(timer_summary, trace_id)
        return {
            "shu": int(res),
            "conf": round(float(val), 3),
            "trace_id": trace_id
        }
    
        
    def detectMachineAttr(self, image_data: bytes, machine_info: str = "", trace_id: str = "", save_result: bool = False,):
        logger.info('start, trace_id: %s, save_result: %s, machine_info: %s', trace_id, save_result, machine_info,)
        show_item = ['food_exist', 'container_type', 'accessory_type', 'rack_level']
        final_res = {}
        save_result = (
            (save_result or settings.paths.save_images_enabled)
            and is_uuid(trace_id)
            and (machine_info or "").startswith(tuple(self.machine_range))
        )

        timer_tools = timerUtils.timer_tool()
        result = self.machine.detect(image_data, machine_info)
        timer_tools.mark('machine_attr')
        for item in show_item:
            if item in result:
                final_res[item] = result[item]

        show_string = " | ".join( f"{k}={v}" for k, v in final_res.items())
        time_string = timer_tools.summary()
        logger.info("检测结果：%s", show_string)
        logger.info("耗时：%s", time_string)
        show_text(show_string, trace_id)
        show_text(time_string, trace_id)
        
        return final_res
        
        

    def detect(self, image_data: bytes, machine_info: str = "", trace_id: str = "", save_result: bool = False):
        logger.info('start, trace_id: %s, save_result: %s, save_env: %s, machine_info: %s', trace_id, save_result, settings.paths.save_images_enabled, machine_info)
        save_result = (
            (save_result or settings.paths.save_images_enabled)
            and is_uuid(trace_id)
            and (machine_info or "").startswith(tuple(self.machine_range))
        )
        timer_tools = timerUtils.timer_tool()
        
        img_bgr = None
        if isinstance(image_data, bytes):
            buf = np.frombuffer(image_data, dtype=np.uint8)
            img_bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        
        mask, _ = self.segmentor.detect(img_bgr, save_result)
        timer_tools.mark('segmentor')
        res = self.roi_extractor.detect(img_bgr, mask, save_result)
        timer_tools.mark('roi_extractor')
        
        temp_data,  _  = self.shengShu.isShu(img_bgr)
        logger.info("isShu: %d", temp_data)
        clip_res = self.single_detect_tree(res, save_result)
        clip_list = clip_res['clip_res']
        final_data = [item[0] for item in clip_list]
        timer_tools.mark('clip_dtct')
        #==========================================================================
        res_text = " ".join(final_data)
        logger.info('res_text: %s', res_text)
        show_text(res_text, trace_id)
        timer_summary = timer_tools.summary()
        logger.info(timer_summary)
        show_text(timer_summary, trace_id)
        save_datas = {}
        if save_result:
            save_datas = self._save_detection_result(clip_res, img_bgr)
        return final_data, save_datas
    
    
    def detect_batch(self, image_dir, save_result: bool = False):
        logger.info('进入detect_batch函数， 路径：%s！', image_dir)
        image_pathList, dir_image_List = utils.get_all_image_paths(image_dir, None, None, ratio=1, min_select=3)
        output_path = 'output'
        src_path = os.path.join(output_path, 'src')
        vis_path = os.path.join(output_path, 'vis')
        lbl_path = os.path.join(output_path, 'lbl')
        crp_path = os.path.join(output_path, 'crp')
        os.makedirs(src_path, exist_ok=True)
        os.makedirs(vis_path, exist_ok=True)
        os.makedirs(lbl_path, exist_ok=True)
        os.makedirs(crp_path, exist_ok=True)
        if len(image_pathList) == 0:
            logger.info('未找到图片！')
            return None
        for image_path in image_pathList:
            time.sleep(0.2)
            try:
                if not os.path.exists(image_path):
                    logger.info('图片不存在, %s', image_path)
                    continue
                foldername = os.path.basename(os.path.dirname(image_path))
                base_name, _ = os.path.splitext(os.path.basename(image_path))
                src_foldername = os.path.join(src_path, foldername)
                vis_foldername = os.path.join(vis_path, foldername)
                lbl_foldername = os.path.join(lbl_path, foldername)
                crp_foldername = os.path.join(crp_path, foldername)
                os.makedirs(src_foldername, exist_ok=True)
                os.makedirs(vis_foldername, exist_ok=True)
                os.makedirs(lbl_foldername, exist_ok=True)
                os.makedirs(crp_foldername, exist_ok=True)
                with open(image_path, 'rb') as f:
                    image_data = f.read()
                    res, draw_data = self.detect(image_data, base_name, "test", save_result)
                    if save_result:               
                        aim_path = f'{src_foldername}/{base_name}.jpg'
                        shutil.copy(image_path, aim_path)
                        
                        rgb_image = cv2.cvtColor(draw_data['vis'], cv2.COLOR_BGR2RGB)
                        pil_image = Image.fromarray(rgb_image)
                        pil_image.save(f'{vis_foldername}/{base_name}.jpg')
                        
                        lbl_image = draw_data['lbl']
                        if len(lbl_image.shape) == 3:
                            lbl_image = cv2.cvtColor(lbl_image, cv2.COLOR_BGR2GRAY)
                        lbl_image = np.clip(lbl_image.astype(np.float32) * 125, 0, 255).astype(np.uint8)
                        pil_image = Image.fromarray(lbl_image)
                        pil_image.save(f'{lbl_foldername}/{base_name}.jpg')
                        
                        crp_image = draw_data['crop']
                        if len(crp_image.shape) == 3:
                            crp_image = cv2.cvtColor(crp_image, cv2.COLOR_BGR2RGB)
                        pil_image = Image.fromarray(crp_image)
                        pil_image.save(f'{crp_foldername}/{base_name}.jpg')
            except Exception as e:
                logger.error("处理文件 %s 时出错：%s\n%s", image_path, str(e), traceback.format_exc())
        logger.info('批量检测完成！')



if __name__ == '__main__':
    from app.core.config.store import get_cfg
    from app.core.observability.logging import setup_logger
    
    setup_logger()
    config = get_cfg()

    image_path = '/data/wangzhuo/01-code/02-project/mainfooddetect/config/images/segment_0902_0892.jpg'
    with open(image_path, 'rb') as f:
        image_data = f.read()
        food_service = FoodService(config)
        res = food_service.detect(image_data, 'test', False)
        logger.info('res: %s', res)

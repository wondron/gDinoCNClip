import os, cv2, time, json
import logging
import sys
from pathlib import Path
import numpy as np
from typing import Dict, Any, List
from PIL import Image

if __name__ == "__main__" and not __package__:
    # 直接运行脚本时，将项目根目录加入模块搜索路径。
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from app.core.observability.log_helper import show_text
from app.domain.common.image_utils import box2xyxy
from app.domain.pipeline.a_gdinoDtct import create_session
import app.domain.common.tokenizer as tokenizer

logger = logging.getLogger("clip")

class ClipDetect:
    def __init__(self, config_data: Dict[str, Any]):
        self._load_config(config_data)


    def _load_config(self, config: Dict[str, Any]):
        clip_cfg = config.get("CLIP_CONFIG", {})
        self.model_path = clip_cfg.get('imag_model_path', None)
        self.text_model_path = clip_cfg.get('text_model_path', None)
        self.device_id = config.get("TOTAL_CONFIG", {}).get("device_id", 0)
        self.threshold = clip_cfg.get('threshold', 0.02)
        self.logit_scale = clip_cfg.get('logit_scale', 100)
        self.test_image_path = clip_cfg.get('test_image_path', "")
        self.recipe_range = clip_cfg.get('recipe_range', [])
        self.common_range = clip_cfg.get('common_range', [])
        self.food_range = clip_cfg.get('food_range', [])
        logger.info(f"菜谱检测范围: {self.recipe_range}")
        logger.info(f"通用检测范围：{self.common_range}")
        logger.info(f"食材检测范围：{self.food_range}")
        show_text(f"菜谱检测范围: {self.recipe_range}")
        show_text(f"通用检测范围：{self.common_range}")
        show_text(f"食材检测范围：{self.food_range}")
        self.init_model()

    def isRangeInFeatures(self, FoodList:List):
        if len(FoodList) == 0:
            logger.warning("未检测到需要检测的范围食材！")
            return False

        known = {**self.food_feat_dict, **self.common_feat_dict, **self.recipe_feat_dict}
        exclude_range = []
        for food in FoodList:
            if food not in known:
                exclude_range.append(food)
        if len(exclude_range) > 0:
            show_msg = "、".join(exclude_range)
            logger.warning(f"{show_msg}, 不在特征库中！")
            return False
        return True


    def warmup(self, runs: int = 3) -> None:
        t0 = time.time()
        dummy = np.zeros((self.shape[2], self.shape[3], self.shape[1]), dtype=np.uint8)
        for _ in range(max(1, runs)):
            self.detect(dummy, "皮皮虾")
        logger.info("预热完成, 次数: %d, 耗时: %.2fs", runs, time.time() - t0)


    def init_model(self):
        self.ini_flag = False
        try:
            if not (self.load_image_model() and self.load_text_model()):
                logger.error("CLIP模型初始化失败！")
                return
            self.food_feat_dict = self.load_text_feature(self.food_range)
            self.common_feat_dict = self.load_text_feature(self.common_range)
            self.recipe_feat_dict = self.load_text_feature(self.recipe_range)
            for name, feat_dict, text_range in (("food_range", self.food_feat_dict, self.food_range),
                                                ("common_range", self.common_feat_dict, self.common_range),
                                                ("recipe_range", self.recipe_feat_dict, self.recipe_range)):
                if len(feat_dict) != len(set(text_range)):
                    logger.error(f"CLIP模型初始化失败！{name} 的文本特征数量与配置不一致")
                    return

            image_bgr = cv2.imread(self.test_image_path)
            if image_bgr is None:
                logger.error(f"CLIP模型初始化失败！无法读取自检图片：{self.test_image_path}")
                return
            self.ini_flag = True
            if self.detect(image_bgr, "皮皮虾") <= 0:
                self.ini_flag = False
                logger.error('CLIP模型初始化失败！输入皮皮虾未和图像特征匹配！')
                return
            logger.info('CLIP模型初始化完成！providers=%s', self.sess.get_providers())
        except Exception as e:
            self.ini_flag = False
            logger.error(f"CLIP模型初始化失败！{e}")
            show_text(f"CLIP模型初始化失败！{e}")



    def load_text_feature(self, text_list):
        try:
            nameList = text_list
            texts = {}
            if not len(nameList):
                return texts
            nameList, text_featureList = self.extract_text_feature(nameList)
            if len(text_featureList) == 0 or len(text_featureList) != len(nameList):
                logger.error("未检测到需要检测的范围食材特征！")
                return texts
            for idx, feature in enumerate(text_featureList):
                texts[nameList[idx]] = feature
            return texts
        except Exception as e:
            logger.error(f"加载CLIP文本特征文件失败！{e}")
            return {}

    def load_text_model(self):
        try:
            if not os.path.exists(self.text_model_path):
                logger.error(f"CLIP 文本模型文件：{self.text_model_path} 不存在！")
                return False
            self.text_sess = create_session(self.text_model_path, "cuda", device_id=self.device_id)
            self.text_input_name = self.text_sess.get_inputs()[0].name
            self.text_output_name = self.text_sess.get_outputs()[0].name
            self.text_shape = self.text_sess.get_inputs()[0].shape
            logger.info(f'clip text 输入名：{self.text_input_name}')
            logger.info(f'clip text 输出名：{self.text_output_name}')
            return True
        except Exception as e:
            logger.error(f"加载CLIP文本模型失败！{e}")
            return False

    def load_image_model(self):
        try:
            if not os.path.exists(self.model_path):
                logger.error(f"CLIP 模型文件：{self.model_path} 不存在！")
                return False
            self.sess = create_session(self.model_path, "cuda", device_id=self.device_id)
            self.input_name = self.sess.get_inputs()[0].name
            self.output_name = self.sess.get_outputs()[0].name
            self.shape = self.sess.get_inputs()[0].shape
            logger.info(f'clip imgs 输入名：{self.input_name}')
            logger.info(f'clip imgs 输出名：{self.output_name}')
            return True
        except Exception as e:
            logger.error(f"加载CLIP模型失败！{e}")
            return False


    def preprocess_image(self, img_bgr):
        try:
            if not isinstance(img_bgr, np.ndarray) or img_bgr.size == 0:
                raise ValueError("输入图像必须为非空 BGR 数组")
            if img_bgr.ndim != 3 or img_bgr.shape[2] != 3:
                raise ValueError("输入图像必须为 HxWx3 BGR 数组")
            img = cv2.resize(img_bgr, (self.shape[2], self.shape[3]), interpolation=cv2.INTER_CUBIC)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.astype(np.float32) / 255.0
            mean = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
            std  = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
            img = (img - mean) / std  # [H, W, 3]
            img = img.transpose(2, 0, 1)  # [3, H, W]
            img = np.expand_dims(img, 0).astype(np.float32)  # [1, 3, H, W]
            return img
        except Exception as e:
            logger.error(f"图片处理失败！{e}")
            return None


    def extract_image_feature(self, img: np.ndarray):
        try:
            if not self.ini_flag:
                logger.error("初始化未成功, 无法就行图像特征提取！")
                return None
            img = self.preprocess_image(img)
            if img is None:
                return None
            feats = self.sess.run([self.output_name], {self.input_name: img})[0]
            feats = feats / (np.linalg.norm(feats, axis=-1, keepdims=True) + 1e-7)
            return feats
        except Exception as e:
            logger.error(f"CLIP模型输入图片处理失败！{e}")
            return None

    def extract_text_feature(self, text_list):
        empty = np.empty((0, 0), dtype=np.float32)
        if not text_list:
            logger.error("给定 text_list 中没有任何合法文本，返回空结果。")
            return [], empty

        seq_len = self.text_shape[1] if len(self.text_shape) > 1 else 52
        tokens_list = tokenizer.tokenize(text_list, context_length=seq_len) 
        tokens = np.array(tokens_list, dtype=np.int64)  # [N, seq_len]
        batch_size = tokens.shape[0]
        expected_bs = self.text_shape[0]  # 可能是 1 / 'batch' / None
        try:
            if not isinstance(expected_bs, int) or expected_bs in (0, -1):
                output = self.text_sess.run([self.text_output_name], {self.text_input_name: tokens})[0]  # [N, dim]
            elif isinstance(expected_bs, int) and expected_bs == batch_size:
                output = self.text_sess.run([self.text_output_name], {self.text_input_name: tokens})[0]  # [N, dim]
            elif isinstance(expected_bs, int) and expected_bs == 1 and batch_size > 1:
                feat_list = []
                for i in range(batch_size):
                    single_tokens = tokens[i:i+1]  # [1, seq_len]
                    single_out = self.text_sess.run([self.text_output_name], {self.text_input_name: single_tokens})[0]  # [1, dim]
                    feat_list.append(single_out[0])  # 取掉 batch 维
                output = np.stack(feat_list, axis=0)  # [N, dim]
            else:
                logger.error(f"CLIP 文本模型 batch 维不匹配：模型期望 {expected_bs}，但输入是 {batch_size}。")
                return [], empty
        except Exception as e:
            logger.error(f"CLIP 文本模型推理失败: {e}")
            return [], empty
        output = output / (np.linalg.norm(output, axis=-1, keepdims=True) + 1e-7)
        return text_list, output



    def batch_detect(self, image, boxes, clip_caption, clip_thre):
        """裁剪归一化 cxcywh 框，返回匹配框的像素 xyxy 坐标及概率。"""
        if not self.ini_flag:
            logger.error("CLIP模型未初始化！")
            return [], []

        try:
            boxes = np.asarray(boxes, dtype=np.float32)
            if boxes.size == 0:
                return [], []
            if boxes.ndim == 1 and boxes.shape == (4,):
                boxes = boxes.reshape(1, 4)
            if boxes.ndim != 2 or boxes.shape[1] != 4:
                raise ValueError("boxes 必须是形状为 (N, 4) 的归一化 cxcywh 坐标")
            valid = np.isfinite(boxes).all(axis=1) & (boxes[:, 2:] > 0).all(axis=1)
            boxes = boxes[valid]
            if len(boxes) == 0:
                return [], []
            if not isinstance(image, Image.Image):
                raise TypeError("image 必须是 PIL 图片")
            # 候选文本特征每个请求只准备一次，所有框共用。
            candidates = self.text_candidates(clip_caption)
            if candidates is None:
                return [], []
            image = image.convert("RGB")
            width, height = image.size

            filtered_rects, confidences = [], []
            for rect in box2xyxy(image, boxes):
                x1, y1, x2, y2 = rect
                if (x2 <= x1 or y2 <= y1 or x2 <= 0 or y2 <= 0
                        or x1 >= width or y1 >= height):
                    continue
                try:
                    crop_rgb = np.asarray(image.crop(rect))
                    crop_bgr = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR)
                    score = float(self.detect(crop_bgr, clip_caption, candidates))
                    if np.isfinite(score) and score > 0:
                        filtered_rects.append(rect)
                        confidences.append(score)
                except Exception as e:
                    logger.error("CLIP检测框 %s 判断失败：%s", rect, e)
            return filtered_rects, confidences
        except Exception as e:
            show_text(f"CLIP批量检测失败：{e}")
            logger.error("CLIP批量检测失败：%s", e)
            return [], []


    def text_candidates(self, input_name: str):
        if not isinstance(input_name, str) or not input_name.strip():
            logger.error("input_name 必须是非空文本")
            return None
        if input_name in self.recipe_range:
            pool = self.common_feat_dict
        else:
            pool = {**self.food_feat_dict, **self.common_feat_dict}
        names, feats = list(pool), list(pool.values())
        if input_name not in pool:
            feat = self.recipe_feat_dict.get(input_name)
            if feat is None:
                _, new_feat = self.extract_text_feature([input_name])
                if len(new_feat) == 0:
                    return None
                feat = new_feat[0]
            names.append(input_name)
            feats.append(feat)
        return names, np.stack(feats)


    def detect(self, imgbgr: np.ndarray, input_name: str, candidates=None) -> float:
        if not self.ini_flag:
            logger.error("CLIP模型未初始化！")
            return 0.0
        if candidates is None:
            candidates = self.text_candidates(input_name)
            if candidates is None:
                return 0.0

        img_feat = self.extract_image_feature(imgbgr)
        if img_feat is None:
            return 0.0

        names, feats = candidates
        logger.info("CLIP检测输入：%s，候选数量=%d，候选文本=%s", input_name, len(names), names)
        show_text(f"CLIP检测输入：{input_name}，候选数量={len(names)}, {names}")
        scores = feats @ img_feat[0]  # cosine similarities, shape [N]
        best = int(np.argmax(scores))
        logger.info("CLIP识别结果：输入=%s，最高分文本=%s，得分=%.4f", input_name, names[best], scores[best])
        return float(scores[best]) if names[best] == input_name else 0.0


if __name__ == '__main__':
    from app.core.config.loader import load_config
    from app.core.config.settings import local_path
    from app.core.observability.logging import setup_logger
    setup_logger()
    config = load_config()
    clip = ClipDetect(config)
    test_path = r"D:\gDinoCNClip\config\images\test.jpg"
    img = cv2.imread(str(test_path))
    if img is None:
        raise ValueError(f"无法读取测试图片：{test_path}")
    if not clip.ini_flag:
        raise RuntimeError("CLIP 模型初始化失败，请检查上方日志和 CLIP_CONFIG 配置")
    
    res = clip.detect(img, input_name="豆腐")
    print(res)

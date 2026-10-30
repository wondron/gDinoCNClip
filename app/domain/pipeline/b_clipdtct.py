import os, cv2, time, json
import logging
import numpy as np
import onnxruntime as ort
from typing import Dict, Optional, Tuple, Any, List
import app.domain.common.tokenizer as tokenizer
import app.domain.common.image_utils as z_tools

logger = logging.getLogger("clip")

class ClipDetect:
    def __init__(self, config_data: Dict[str, Any]):
        self._load_config(config_data)
        self.init_model()


    def _load_config(self, config: Dict[str, Any]):
        clip_cfg = config.get("CLIP_CONFIG", {})
        self.model_path = clip_cfg.get('model_path', None)
        self.text_model_path = clip_cfg.get('text_model_path', None)
        self.text_path = clip_cfg.get('text_path', None)
        self.threshold = clip_cfg.get('threshold', 0.02)
        self.topk = clip_cfg.get('topk', 5)


    def isRangeInFeatures(self, FoodList:List):
        if len(FoodList) == 0:
            logger.warning("未检测到需要检测的范围食材！")
            return False

        exclude_range = []
        for food in FoodList:
            if food not in self.texts:
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
            self.detect(dummy)
        logger.info("预热完成, 次数: %d, 耗时: %.2fs", runs, time.time() - t0)
    

    def get_providers(self):
        available = ort.get_available_providers()
        providers = []
        if "CUDAExecutionProvider" in available:
            providers.append("CUDAExecutionProvider")
        if "CPUExecutionProvider" in available:
            providers.append("CPUExecutionProvider")
        return providers or ["CPUExecutionProvider"]
        # return ["CPUExecutionProvider"]


    def init_model(self):
        self.ini_flag = False
        textflag = self.load_text_feature()
        imagflag = self.load_image_model()
        textmdlflag = self.load_text_model()
        textRangeflag = self.init_textRange_with_text_model()
        self.ini_flag = textflag and imagflag and textmdlflag and textRangeflag

        if not self.ini_flag:
            logger.error("CLIP模型初始化失败！")
            return
        check_text_feature = self.debug_compare_text_features()
        self.ini_flag = check_text_feature
        if self.ini_flag:
            logger.info('CLIP模型初始化完成！')
    

    def init_textRange_with_text_model(self):
        try:
            self.feats_with_text_model = {}

            for text in self.texts:
                _, feat = self.extract_text_feature_with_model([text])
                if feat.shape[0] == 1:
                    self.feats_with_text_model[text] = feat[0]
                else:
                    logger.error(f"使用文本模型提取文本特征失败: {text}")
                
            logger.info(f"使用文本模型提取文本特征完成，数量：{len(self.feats_with_text_model)}")
            return True
        except Exception as e:
            logger.error(f"使用文本模型提取文本特征失败！{e}")
            return False


    def load_text_feature(self):
        try:
            if not os.path.exists(self.text_path):
                logger.error(f"CLIP 文本特征文件：{self.text_path} 不存在！")
                return False
            
            self.texts = []
            self.feats = []
            self.logit_scale = None
            with open(self.text_path, "r", encoding="utf-8") as f:
                for line in f:
                    entry = json.loads(line.strip())
                    self.texts.append(entry["text"])
                    self.feats.append(entry["feature"])
                    if self.logit_scale is None and "logit_scale" in entry:
                        self.logit_scale = float(entry["logit_scale"])
            
            self.feats = np.asarray(self.feats, dtype=np.float32)
            self.feats = self.feats / (np.linalg.norm(self.feats, axis=-1, keepdims=True) + 1e-7)
            if self.logit_scale is None:
                self.logit_scale = 1.0

            logger.info(f"类别数量：{len(self.texts)}")
            logger.info(f"文本特征维度：{self.feats.shape[1]}")
            logger.info(f"logit_scale: {self.logit_scale:.4f}")
            return True
        except Exception as e:
            logger.error(f"加载CLIP文本特征文件失败！{e}")
            return False


    def load_text_model(self):
        try:
            if not os.path.exists(self.text_model_path):
                logger.error(f"CLIP 文本模型文件：{self.text_model_path} 不存在！")
                return False
            providers = self.get_providers()
            sess_opts = ort.SessionOptions()
            sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self.text_sess = ort.InferenceSession(str(self.text_model_path), sess_opts, providers=providers)
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

            providers = self.get_providers()
            sess_opts = ort.SessionOptions()
            sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self.sess = ort.InferenceSession(str(self.model_path), sess_opts, providers=providers)

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
            feats = self.sess.run([self.output_name], {self.input_name: img})[0]
            feats = feats / (np.linalg.norm(feats, axis=-1, keepdims=True) + 1e-7)
            return feats
        except Exception as e:
            logger.error(f"CLIP模型输入图片处理失败！{e}")
            return None


    def extract_text_feature(self, text_list=None):
        idx_list = []
        feat_list = []
        text_features = None

        if text_list is None or len(text_list) == 0:
            idx_list = list(range(len(self.texts)))
            text_features = self.feats
            return idx_list, text_features
        
        for text in text_list:
            if text not in self.texts:
                logger.error(f"文本：{text} 不在 CLIP 模型文本列表中！")
                continue
            idx = self.texts.index(text)
            idx_list.append(idx)
            feat_list.append(self.feats[idx])
        if not idx_list:
            logger.error("给定 text_list 中没有任何合法文本，返回空结果。")
            return [], np.empty((0, self.feats.shape[1]), dtype=self.feats.dtype)
        text_features = np.stack(feat_list, axis=0)  # [N, dim]
        return idx_list, text_features


    def extract_text_feature_with_model(self, text_list=None):
        """
        使用 ONNX 文本模型提取文本特征。
        返回 (idx_list, text_features)，结构与 extract_text_feature 完全一致。
        """
        if text_list is None or len(text_list) == 0:
            text_list = self.texts

        if not text_list:
            logger.error("给定 text_list 中没有任何合法文本，返回空结果。")
            return [], np.empty((0, self.feats.shape[1]), dtype=self.feats.dtype)

        seq_len = self.text_shape[1] if len(self.text_shape) > 1 else 52
        tokens_list = tokenizer.tokenize(text_list, context_length=seq_len) 
        tokens = np.array(tokens_list, dtype=np.int64)  # [N, seq_len]
        batch_size = tokens.shape[0]
        expected_bs = self.text_shape[0]  # 可能是 1 / 'batch' / None
        try:
            if not isinstance(expected_bs, int) or expected_bs in (0, -1):
                output = self.text_sess.run(
                    [self.text_output_name],
                    {self.text_input_name: tokens}
                )[0]  # [N, dim]

            elif isinstance(expected_bs, int) and expected_bs == batch_size:
                output = self.text_sess.run(
                    [self.text_output_name],
                    {self.text_input_name: tokens}
                )[0]  # [N, dim]

            elif isinstance(expected_bs, int) and expected_bs == 1 and batch_size > 1:
                feat_list = []
                for i in range(batch_size):
                    single_tokens = tokens[i:i+1]  # [1, seq_len]
                    single_out = self.text_sess.run(
                        [self.text_output_name],
                        {self.text_input_name: single_tokens}
                    )[0]  # [1, dim]
                    feat_list.append(single_out[0])  # 取掉 batch 维

                output = np.stack(feat_list, axis=0)  # [N, dim]

            else:
                logger.error(
                    f"CLIP 文本模型 batch 维不匹配：模型期望 {expected_bs}，但输入是 {batch_size}。"
                )
                return [], np.empty((0, self.feats.shape[1]), dtype=self.feats.dtype)

        except Exception as e:
            logger.error(f"CLIP 文本模型推理失败: {e}")
            return [], np.empty((0, self.feats.shape[1]), dtype=self.feats.dtype)
        output = output / (np.linalg.norm(output, axis=-1, keepdims=True) + 1e-7)
        return text_list, output



    def detect_with_text_model(self, img: np.ndarray, text_range = None) -> List[Tuple[str, float]]:
        if not self.ini_flag:
            logger.error("CLIP模型未初始化！")
            return []
        try:
            img_feat = self.extract_image_feature(img)
            if img_feat.ndim == 2:
                img_feat = img_feat[0]  # [dim]

            if text_range == list(self.feats_with_text_model.keys()):
                text_List = text_range
                text_feats = np.stack(
                    [self.feats_with_text_model[t] for t in text_range],
                    axis=0
                )  # [N, dim]
            else:
                text_List, text_feats = self.extract_text_feature_with_model(text_range)  # text_feats: [N, dim]

            if text_feats.shape[0] == 0:
                logger.error("没有可用的文本特征，退出 detect。")
                return []
            
            sims = text_feats @ img_feat
            if self.logit_scale is not None:
                logits = sims * float(self.logit_scale)  # [N]
                #softmax
                max_logit = np.max(logits)
                exp_x = np.exp(logits - max_logit)
                probs = exp_x / (exp_x.sum() + 1e-8)
                scores = probs  # 用概率作为 score
            else:
                scores = sims  # 没有温度系数，就用相似度作为 score
            N = scores.shape[0]
            k = min(self.topk, N)
            top_idx_local = np.argsort(-scores)[:k]
            results: List[Tuple[str, float]] = []
            for local_i in top_idx_local:
                text = text_List[local_i]
                score = float(scores[local_i])
                if score >= self.threshold:
                    results.append((text, score))
                else:
                    logger.info(f"相似度低于阈值 {self.threshold}，忽略：{text} | {score}")
            
            logger.info(f"CLIP模型识别结果：{results}")
            return results
        except Exception as e:
            logger.error(f"CLIP模型输入图片处理失败！{e}")
            return []


    def debug_compare_text_features(self, text_list=None):
        idx_old, feat_old = self.extract_text_feature(text_list)
        idx_new, feat_new = self.extract_text_feature_with_model(text_list)  # [N, dim]

        if len(idx_old) == 0 or len(idx_new) == 0:
            logger.error("没有拿到有效的文本特征，无法对比。")
            return False

        if len(idx_old) != len(idx_new):
            logger.warning(f"idx_old 和 idx_new 长度不一致！idx_old={len(idx_old)}, idx_new={len(idx_new)}")
            return False

        if feat_old.shape != feat_new.shape:
            logger.error(f"特征 shape 不一致：old={feat_old.shape}, new={feat_new.shape}")
            return False

        cos_sim = np.sum(feat_old * feat_new, axis=1)
        diff = feat_old - feat_new
        diff_norm = np.linalg.norm(diff, axis=1)

        logger.info(f"文本特征对比：样本数 N = {feat_old.shape[0]}")
        logger.info(
            f"cos-sim 统计：mean={cos_sim.mean():.6f}, "
            f"min={cos_sim.min():.6f}, max={cos_sim.max():.6f}"
        )
        logger.info(
            f"L2 diff 统计：mean={diff_norm.mean():.6f}, "
            f"min={diff_norm.min():.6f}, max={diff_norm.max():.6f}"
        )
        for i in range(min(5, len(idx_old))):
            t = self.texts[idx_old[i]]
            logger.info(f"文本: {t} | cos-sim={cos_sim[i]:.6f}, diff_norm={diff_norm[i]:.6f}")

        if cos_sim.mean() < 0.995 and diff_norm.mean() > 1e-3:
            logger.error("文本特征和模型提取的特征不一致！")
            return False

        return True



    def detect(self, imgbgr: np.ndarray, text_range= None) -> List[Tuple[str, float]]:
        if not self.ini_flag:
            logger.error("CLIP模型未初始化！")
            return []

        try:
            img_feat = self.extract_image_feature(imgbgr)
            if img_feat.ndim == 2:
                img_feat = img_feat[0]  # [dim]

            idx_list, text_feats = self.extract_text_feature(text_range)  # text_feats: [N, dim]
            if text_feats.shape[0] == 0:
                logger.error("没有可用的文本特征，退出 detect。")
                return []
            sims = text_feats @ img_feat
            if self.logit_scale is not None:
                logits = sims * float(self.logit_scale)  # [N]
                #softmax
                max_logit = np.max(logits)
                exp_x = np.exp(logits - max_logit)
                probs = exp_x / (exp_x.sum() + 1e-8)
                scores = probs
            else:
                scores = sims
            N = scores.shape[0]
            k = min(self.topk, N)
            top_idx_local = np.argsort(-scores)[:k]
            results: List[Tuple[str, float]] = []
            for local_i in top_idx_local:
                global_idx = idx_list[local_i]
                text = self.texts[global_idx]
                score = float(scores[local_i])
                if score >= self.threshold:
                    results.append((text, score))
            
            logger.info(f"CLIP模型识别结果：{results}")
            return results
        except Exception as e:
            logger.error(f"CLIP模型输入图片处理失败！{e}")
            return []


if __name__ == '__main__':
    from app.core.config.store import get_cfg
    from app.core.observability.logging import setup_logger
    
    setup_logger()
    config = get_cfg()
    img = cv2.imread("/data/wangzhuo/01-code/02-project/mainfooddetect/123123.jpg")

    clip = ClipDetect(config)
    rangeList = ['速冻包子', '手工包子', '无食材', '带鱼', '大虾', '秋刀鱼', '可乐', '雪碧', '牛奶', '豆浆', '鸡蛋', '火腿肠', '昌鱼', '鳕鱼', '三文鱼', '草鱼', '鲈鱼', '鲫鱼', '黄花鱼', '带鱼段', '龙利鱼', '巴沙鱼', '鳝鱼段', '虾仁', '虾球', '虾滑', '冻虾', '冻虾仁']
    res = clip.detect_with_text_model(img, text_range= rangeList)
    print(res)

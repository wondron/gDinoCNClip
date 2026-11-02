from app.domain.pipeline.a_gdinoDtct import GdinoDetect
from app.domain.pipeline.b_clipdtct import ClipDetect
from app.domain.common.image_utils import box2xyxy
from app.domain.common.timer_utils import timer_tool
from app.core.observability.log_helper import show_text
import logging
import io, time
import math
from PIL import Image, ImageDraw


logger = logging.getLogger("food_service")


class Detector:
    def __init__(self, config_dict, dino=None, clip=None):
        self.load_config(config_dict, dino, clip)
        
    def load_config(self, config_dict, dino=None, clip=None):
        # 传入已加载的组件时直接复用，配置热更新只重新加载发生变化的模型。
        self.dino = GdinoDetect(config_dict) if dino is None else dino
        self.clip = ClipDetect(config_dict) if clip is None else clip
        
    def gDino_filter(self, image_cv, boxes, parases, promt):
        matching_indices = [i for i, label in enumerate(parases) if promt in label]
        matching_boxes = boxes[matching_indices]
        xyxy =box2xyxy(image_cv, matching_boxes)
        confidences = [0.6] * len(matching_boxes)        
        return xyxy, confidences
        
    def detect(self, image_data, dino_caption, dino_box_thre, dino_text_thre, clip_caption, clip_thre):
        try:
            timer = timer_tool()
            with Image.open(io.BytesIO(image_data)) as source:
                image = source.convert("RGB")
            timer.mark("load image")
            boxes, _, parases = self.dino.detect(image, dino_caption, dino_box_thre, dino_text_thre) 
            timer.mark("dino detect")
            if clip_thre <= 0:
                rects, confident = self.gDino_filter(image, boxes, parases, dino_caption)
                return rects, confident
            rects, confident = self.clip.batch_detect(image, boxes, clip_caption, clip_thre)  
            timer.mark("clip detect")
            logging.info(f"{timer.summary()}")
            show_text(timer.summary())
            return rects, confident
        except Exception as e:
            print(f"An error occurred: {e}")
            return [], []

    def draw_result(self, image_data, boxes, confidents):
        """在图片字节或 PIL 图片上绘制像素 xyxy 框和置信度，返回新的 RGB 图片。"""
        if len(boxes) != len(confidents):
            raise ValueError("boxes and confidents must have the same length")
        if isinstance(image_data, Image.Image):
            image = image_data.convert("RGB")
        else:
            with Image.open(io.BytesIO(image_data)) as source:
                image = source.convert("RGB")

        draw = ImageDraw.Draw(image)
        width, height = image.size
        for box, confidence in zip(boxes, confidents):
            x1, y1, x2, y2 = map(float, box)
            confidence = float(confidence)
            if not all(math.isfinite(value) for value in (x1, y1, x2, y2, confidence)):
                raise ValueError("Box coordinates and confidence must be finite")
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width - 1, x2), min(height - 1, y2)
            if x2 <= x1 or y2 <= y1:
                continue

            draw.rectangle((x1, y1, x2, y2), outline="red", width=3)
            label = f"{confidence:.2f}"
            left, top, right, bottom = draw.textbbox((0, 0), label)
            label_width, label_height = right - left + 4, bottom - top + 4
            label_x = min(x1, max(0, width - label_width))
            label_y = max(0, y1 - label_height)
            draw.rectangle((label_x, label_y, label_x + label_width,
                            label_y + label_height), fill="red")
            draw.text((label_x + 2 - left, label_y + 2 - top), label, fill="white")
        return image

    
if __name__ == '__main__':
    detector = Detector('config/param/config.yaml')
    image_path = 'config/images/test.jpg'
    image_data = open(image_path, 'rb').read()
    start_time = time.time()  # 记录开始时间
    rect, confident = detector.detect(image_data, 'food', 0.3, 0.4, '食物', 0.5)
    end_time = time.time()  # 记录结束时间
    detector.draw_result(image_data, rect, confident).save('draw_rect.jpg')
    print(f"检测耗时: {end_time - start_time:.2f}秒")  # 输出检测时间，保留两位小数
    print('rect:', rect, '\nconf:', confident)

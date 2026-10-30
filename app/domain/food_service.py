from app.domain.pipeline.a_gdinoDtct import GdinoDetect
from app.domain.pipeline.b_clipdtct import ClipDetect
from app.domain.common.image_utils import box2xyxy
import io, time
from PIL import Image



class Detector:
    def __init__(self, config_path):
        self.load_config(config_path)
        
    def load_config(self, config_path):
        self.dino = GdinoDetect(config_path)
        self.clip = ClipDetect(config_path)
        
    def gDino_filter(self, image_cv, boxes, parases, promt):
        matching_indices = [i for i, label in enumerate(parases) if promt in label]
        matching_boxes = boxes[matching_indices]
        xyxy =box2xyxy(image_cv, matching_boxes)
        confidences = [0.6] * len(matching_boxes)        
        return xyxy, confidences
        
    def detect(self, image_data, dino_caption, dino_box_thre, dino_text_thre, clip_caption, clip_thre):
        try:
            image = Image.open(io.BytesIO(image_data)).convert("RGB")
            boxes, _, parases = self.dino.detect(image, dino_caption, dino_box_thre, dino_text_thre) 
            if clip_thre <= 0:
                rects, confident = self.gDino_filter(image, boxes, parases, dino_caption)
                return rects, confident
            rects, confident = self.clip.batch_detect(image, boxes, clip_caption, clip_thre)  
            return rects, confident
        except Exception as e:
            print(f"An error occurred: {e}")
            return [], []

    
if __name__ == '__main__':
    detector = Detector('config/config.yaml')
    image_path = 'images/dddd.jpg'
    image_data = open(image_path, 'rb').read()
    start_time = time.time()  # 记录开始时间
    rect, confident = detector.detect(image_data, 'food', 0.3, 0.4, '', 10)
    end_time = time.time()  # 记录结束时间
    # 将检测结果可视化
    from PIL import ImageDraw
    
    # 打开图片
    img = Image.open(image_path)
    draw = ImageDraw.Draw(img)
    # 遍历所有检测框
    for box in rect:
        # 绘制矩形框
        draw.rectangle(box, outline="red", width=3)
    img.save('draw_rect.jpg')
    print(f"检测耗时: {end_time - start_time:.2f}秒")  # 输出检测时间，保留两位小数
    print('rect:', rect, '\nconf:', confident)
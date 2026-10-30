import json
import logging
from io import BytesIO
from uuid import uuid4

from fastapi import APIRouter, File, Form, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from PIL import Image

from app.core.observability.log_helper import show_text

router = APIRouter()
logger = logging.getLogger("predict")


@router.post("/food_detect/")
async def food_detect(
    request: Request,
    image: UploadFile = File(...),
    dino_cap: str = Form(...),
    dino_box: float = Form(...),
    dino_text: float = Form(...),
    clip_cap: str = Form(...),
    clip_thre: float = Form(...),
):
    if image is None or not image.content_type or not image.content_type.startswith("image/"):
        return JSONResponse(content={"error": "Uploaded file is not an image"}, status_code=400)
    try:
        image_data = await image.read()
        logger.info("收到食材检测请求 | bytes=%d | dino_cap=%r | dino_box=%s | dino_text=%s | clip_cap=%r | clip_thre=%s",
                    len(image_data), dino_cap[:512], dino_box, dino_text, clip_cap[:512], clip_thre)
        service = request.app.state.runtime.food_service
        rects, confidences = await run_in_threadpool(
            service.detect, image_data, dino_cap, dino_box, dino_text, clip_cap, clip_thre,
        )
        # Existing reference clients parse these Python-list strings.
        return JSONResponse(content={"rects": str(rects), "conf": str(confidences)})
    except json.JSONDecodeError:
        return JSONResponse(content={"error": "Invalid rect_list format, must be a valid JSON array."}, status_code=400)
    except Exception as exc:
        logger.exception("食材检测失败")
        show_text("食材检测失败", logger_level=logging.ERROR, exc_info=True)
        return JSONResponse(content={"error": "Unexpected error", "details": str(exc)}, status_code=500)


def save_jpeg(image_bytes, destination, quality):
    with Image.open(BytesIO(image_bytes)) as source:
        if source.mode in ["RGBA", "P", "LA", "PA"]:
            source = source.convert("RGB")
        source.save(destination, format="JPEG", quality=quality)


@router.post("/process/")
async def process_endpoint(
    request: Request,
    image: UploadFile = File(...),
    discri: str = Form(...),
    box_thre: float = Form(...),
    text_thre: float = Form(...),
):
    image_path = None
    runtime = request.app.state.runtime
    settings = runtime.settings
    try:
        if not image.content_type or not image.content_type.startswith("image/"):
            logger.warning("上传内容不是图片 | content_type=%s", image.content_type)
            return JSONResponse(content={"error": "Uploaded file is not an image"}, status_code=400)
        image_path = runtime.upload_dir / f"{uuid4().hex}.jpg"
        image_bytes = await image.read()
        logger.info("收到检测请求 | image=%s | bytes=%d | caption=%r | box_thre=%s | text_thre=%s",
                    image_path.name, len(image_bytes), discri[:512], box_thre, text_thre)
        try:
            await run_in_threadpool(save_jpeg, image_bytes, image_path, settings.paths.jpeg_quality)
        except Exception:
            logger.exception("图片读取或保存失败")
            return JSONResponse(content={"error": "Invalid image format or corrupted image"}, status_code=400)
        try:
            predictor = runtime.predictor
            result = await run_in_threadpool(predictor.predict, image_path, discri, box_thre, text_thre)
        except Exception as exc:
            logger.exception("模型推理失败")
            show_text("模型推理失败", logger_level=logging.ERROR, exc_info=True)
            return JSONResponse(content={"error": "Model inference failed", "details": str(exc)}, status_code=500)
        return JSONResponse(content={"boxes": result["boxes"], "phrases": result["phrases"], "logits": result["logits"]})
    except Exception as exc:
        logger.exception("请求处理失败")
        return JSONResponse(content={"error": "Unexpected error", "details": str(exc)}, status_code=500)
    finally:
        if image_path is not None and not settings.paths.save_images_enabled:
            try:
                image_path.unlink(missing_ok=True)
            except OSError:
                logger.warning("临时图片清理失败 | path=%s", image_path, exc_info=True)

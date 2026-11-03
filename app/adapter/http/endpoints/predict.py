import json
import logging
from io import BytesIO
from uuid import uuid4

from fastapi import APIRouter, File, Form, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from PIL import Image

from app.core.observability.log_helper import show_text
from app.domain.groundingdino.visualize import annotate

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
    save_result: bool = Form(False),
):
    if image is None or not image.content_type or not image.content_type.startswith("image/"):
        return JSONResponse(content={"error": "Uploaded file is not an image"}, status_code=400)
    try:
        image_data = await image.read()
        logger.info("收到食材检测请求 | bytes=%d | dino_cap=%r | dino_box=%s | dino_text=%s | clip_cap=%r | clip_thre=%s",
                    len(image_data), dino_cap[:512], dino_box, dino_text, clip_cap[:512], clip_thre)
        runtime = request.app.state.runtime
        service = runtime.food_service
        rects, confidences = await run_in_threadpool(
            service.detect, image_data, dino_cap, dino_box, dino_text, clip_cap, clip_thre,
        )
        if save_result:
            destination = runtime.result_dir / f"{uuid4().hex}.jpg"
            await run_in_threadpool(save_food_result, service, image_data, rects, confidences,
                                    destination, runtime.settings.paths.jpeg_quality)
            logger.info("食材检测结果图已保存 | path=%s", destination)
        # Existing reference clients parse these Python-list strings.
        return JSONResponse(content={"rects": str(rects), "conf": str(confidences)})
    except json.JSONDecodeError:
        return JSONResponse(content={"error": "Invalid rect_list format, must be a valid JSON array."}, status_code=400)
    except Exception as exc:
        logger.exception("食材检测失败")
        show_text("食材检测失败", logger_level=logging.ERROR, exc_info=True)
        return JSONResponse(content={"error": "Unexpected error", "details": str(exc)}, status_code=500)


def save_food_result(service, image_bytes, rects, confidences, destination, quality):
    with service.draw_result(image_bytes, rects, confidences) as rendered:
        destination.parent.mkdir(parents=True, exist_ok=True)
        rendered.save(destination, format="JPEG", quality=quality)


def decode_jpeg(image_bytes, quality):
    """Preserve the original JPEG round trip without creating an upload file."""
    encoded = BytesIO()
    with Image.open(BytesIO(image_bytes)) as source:
        if source.mode in ["RGBA", "P", "LA", "PA"]:
            with source.convert("RGB") as rgb:
                rgb.save(encoded, format="JPEG", quality=quality)
        else:
            source.save(encoded, format="JPEG", quality=quality)
    encoded.seek(0)
    with Image.open(encoded) as recoded:
        return recoded.convert("RGB")


@router.post("/process/")
async def process_endpoint(
    request: Request,
    image: UploadFile = File(...),
    discri: str = Form(...),
    box_thre: float = Form(...),
    text_thre: float = Form(...),
    save_result: bool = Form(False),
):
    decoded_image = None
    runtime = request.app.state.runtime
    settings = runtime.settings
    try:
        if not image.content_type or not image.content_type.startswith("image/"):
            logger.warning("上传内容不是图片 | content_type=%s", image.content_type)
            return JSONResponse(content={"error": "Uploaded file is not an image"}, status_code=400)
        image_bytes = await image.read()
        logger.info("收到检测请求 | bytes=%d | caption=%r | box_thre=%s | text_thre=%s",
                    len(image_bytes), discri[:512], box_thre, text_thre)
        try:
            decoded_image = await run_in_threadpool(decode_jpeg, image_bytes, settings.paths.jpeg_quality)
        except Exception:
            logger.exception("图片读取或编码失败")
            return JSONResponse(content={"error": "Invalid image format or corrupted image"}, status_code=400)
        try:
            predictor = runtime.predictor
            result = await run_in_threadpool(predictor.predict, decoded_image, discri, box_thre, text_thre)
        except Exception as exc:
            logger.exception("模型推理失败")
            show_text("模型推理失败", logger_level=logging.ERROR, exc_info=True)
            return JSONResponse(content={"error": "Model inference failed", "details": str(exc)}, status_code=500)
        if save_result:
            destination = runtime.result_dir / f"{uuid4().hex}.jpg"
            await run_in_threadpool(annotate, decoded_image, result, destination)
            logger.info("检测结果图已保存 | path=%s", destination)
        return JSONResponse(content={"boxes": result["boxes"], "phrases": result["phrases"], "logits": result["logits"]})
    except Exception as exc:
        logger.exception("请求处理失败")
        return JSONResponse(content={"error": "Unexpected error", "details": str(exc)}, status_code=500)
    finally:
        if decoded_image is not None:
            decoded_image.close()

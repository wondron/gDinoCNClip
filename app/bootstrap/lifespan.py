import logging
from contextlib import asynccontextmanager

from fastapi.concurrency import run_in_threadpool

from app.bootstrap.state import ApplicationState
from app.core.config.nacos import NacosConfigProvider
from app.core.config.settings import local_path
from app.core.observability.log_helper import show_text
from app.core.observability.logging import setup_logger
from app.domain.food_service import Detector

logger = logging.getLogger("main.lifespan")


def _same_config_value(left, right):
    # YAML booleans/floats must not bypass validation by comparing equal to ints.
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(_same_config_value(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(_same_config_value(a, b) for a, b in zip(left, right))
    return left == right


def warmup_models(predictor, app_settings):
    if not app_settings.warmup.enabled:
        return
    cfg = app_settings.warmup
    logger.info("模型预热 | runs=%d | image=%s", cfg.runs, cfg.image_path)
    for _ in range(max(1, cfg.runs)):
        predictor.predict(local_path(cfg.image_path), cfg.caption, predictor.box_threshold, predictor.text_threshold)


def _build_detector(config):
    detector = Detector(config)
    if not detector.clip.ini_flag:
        raise RuntimeError("CLIP model initialization failed")
    return detector


def apply_nacos_config_update(app, snapshot):
    runtime = app.state.runtime
    if runtime.stopping:
        return False
    try:
        predictor = runtime.predictor
        model_changed = not _same_config_value(
            snapshot.config.get("GROUNDINGDINO_CONFIG"), runtime.config.get("GROUNDINGDINO_CONFIG"),
        )
        device_changed = not _same_config_value(
            snapshot.config.get("TOTAL_CONFIG", {}).get("device_id", 0),
            runtime.config.get("TOTAL_CONFIG", {}).get("device_id", 0),
        )
        clip_changed = not _same_config_value(
            snapshot.config.get("CLIP_CONFIG"), runtime.config.get("CLIP_CONFIG"),
        )
        food_service = runtime.food_service
        if food_service is None or model_changed or device_changed or clip_changed:
            food_service = _build_detector(snapshot.config)
            predictor = food_service.dino
            warmup_models(predictor, runtime.settings)
        with runtime.lock:
            if runtime.stopping:
                return False
            runtime.config = snapshot.config
            runtime.config_info = runtime.config_provider.source_info(snapshot)
            runtime.predictor = predictor
            runtime.food_service = food_service
            runtime.reload_error = None
        logger.info("Nacos 配置更新完成 | profile=%s", snapshot.profile["profile"])
        show_text("Nacos 配置更新完成", profile=snapshot.profile["profile"])
        return True
    except Exception as exc:
        runtime.reload_error = f"{type(exc).__name__}: {exc}"
        logger.exception("Nacos 配置更新失败，继续使用已加载模型")
        show_text("Nacos 配置更新失败，继续使用已加载模型", logger_level=logging.ERROR, exc_info=True)
        return False


def create_lifespan(app_settings):
    @asynccontextmanager
    async def lifespan(app):
        setup_logger(app_settings=app_settings)
        provider = NacosConfigProvider(app_settings.nacos)
        runtime = None
        try:
            config, source = await run_in_threadpool(provider.load, app_settings.config.yaml_path)
            runtime = ApplicationState(app_settings, config, source, provider)
            app.state.runtime = runtime
            runtime.upload_dir.mkdir(parents=True, exist_ok=True)
            logger.info("服务启动 | service=%s | config_source=%s | yaml=%s",
                        app_settings.application.service_name, source["source"],
                        local_path(app_settings.config.yaml_path))
            runtime.food_service = await run_in_threadpool(_build_detector, config)
            runtime.predictor = runtime.food_service.dino
            await run_in_threadpool(warmup_models, runtime.predictor, app_settings)
            provider.start_polling(lambda snapshot: apply_nacos_config_update(app, snapshot))
            show_text("服务启动完成", service=app_settings.application.service_name,
                      configSource=source["source"])
            yield
        except Exception:
            logger.exception("服务启动或运行失败")
            show_text("服务启动或运行失败", logger_level=logging.ERROR, exc_info=True)
            raise
        finally:
            if runtime is not None:
                with runtime.lock:
                    runtime.stopping = True
            await run_in_threadpool(provider.close)
            if runtime is not None:
                runtime.predictor = None
                runtime.food_service = None
            logger.info("服务关闭完成")
            show_text("服务关闭完成")
    return lifespan

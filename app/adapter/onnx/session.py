"""Create a checked ONNX Runtime session and configure native logging."""

import logging

from app.adapter.onnx.cuda import _preload_windows_cuda

logger = logging.getLogger("onnx.session")


def create_session(model, provider="cpu", intra_op_num_threads=0, *, device_id=0,
                   log_severity_level=None):
    """Create ORT session; a requested CUDA provider must actually initialize."""
    import onnxruntime as ort

    if log_severity_level is None:
        from app.core.config.constant import ORT_LOG_SEVERITY_LEVEL
        log_severity_level = ORT_LOG_SEVERITY_LEVEL
    if (isinstance(log_severity_level, bool) or not isinstance(log_severity_level, int)
            or not 0 <= log_severity_level <= 4):
        raise ValueError("ORT_LOG_SEVERITY_LEVEL must be an integer from 0 to 4")
    if provider not in ("cpu", "cuda"):
        raise ValueError("provider must be cpu or cuda")
    if intra_op_num_threads < 0 or device_id < 0:
        raise ValueError("threads and device_id must be nonnegative")
    # Graph partition warnings use the session logger; CUDA ScatterND kernel
    # warnings use the default logger. Configure both, preserving errors.
    ort.set_default_logger_severity(log_severity_level)
    requested = "CUDAExecutionProvider" if provider == "cuda" else "CPUExecutionProvider"
    available = ort.get_available_providers()
    logger.info("ONNX Runtime providers | available=%s | requested=%s | device_id=%d | ort_log_severity=%d",
                available, requested, device_id, log_severity_level)
    if requested not in available:
        raise RuntimeError(f"{requested} unavailable; installed ORT providers: {available}")
    if provider == "cuda":
        _preload_windows_cuda()
    options = ort.SessionOptions()
    options.log_severity_level = log_severity_level
    if intra_op_num_threads:
        options.intra_op_num_threads = intra_op_num_threads
    # TF32 can change detection scores around thresholds; prefer FP32 agreement
    # with the reference model when CUDA is explicitly requested.
    providers = [(requested, {"use_tf32": "0", "device_id": str(device_id)})] if provider == "cuda" else [requested]
    session = ort.InferenceSession(str(model), sess_options=options, providers=providers)
    session.disable_fallback()
    if provider == "cuda" and session.get_providers()[0] != requested:
        raise RuntimeError(
            "CUDA provider initialization failed; refusing to run on CPU. "
            f"Active providers: {session.get_providers()}"
        )
    return session

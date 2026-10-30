ARG PYTHON_IMAGE=python:3.11-slim-bookworm
FROM ${PYTHON_IMAGE}

ARG PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

# CUDA 12 / cuDNN 9 are supplied by the pinned NVIDIA wheels in requirements.txt.
# Include the driver paths injected by NVIDIA Container Toolkit at runtime.
ENV LD_LIBRARY_PATH=/usr/local/lib/python3.11/site-packages/nvidia/cuda_runtime/lib:/usr/local/lib/python3.11/site-packages/nvidia/cublas/lib:/usr/local/lib/python3.11/site-packages/nvidia/cudnn/lib:/usr/local/lib/python3.11/site-packages/nvidia/cufft/lib:/usr/local/lib/python3.11/site-packages/nvidia/curand/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64

RUN apt-get -o Acquire::Retries=3 update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        libgomp1 \
        libstdc++6 \
        zlib1g \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Cache dependency installation independently of application/model updates.
COPY requirements.txt ./
RUN python -m pip install \
        --only-binary=:all: \
        --retries 5 \
        --timeout 120 \
        --index-url "${PIP_INDEX_URL}" \
        -r requirements.txt \
    && python -m pip check \
    && python -c "import importlib.util; import onnxruntime as ort; assert 'CUDAExecutionProvider' in ort.get_available_providers(), ort.get_available_providers(); assert importlib.util.find_spec('torch') is None; print('ONNX Runtime:', ort.__version__)"

# Keep the large model layer cached when only application code changes.
COPY config/model/ ./config/model/
COPY app/ ./app/
COPY config/param/ ./config/param/
COPY config/images/ ./config/images/
COPY main.py ./

RUN mkdir -p /app/output

EXPOSE 8678

HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8678/openapi.json', timeout=3).close()"]

CMD ["python", "main.py"]

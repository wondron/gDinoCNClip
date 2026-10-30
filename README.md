# GroundingDINO ONNX 服务

沿用 `Image-Embedding/backend` 的配置、日志和模型目录约定，按启动、HTTP、检测流程和算法辅助工具组织。运行时不需要 PyTorch、GroundingDINO 源码或 `_C` 扩展。

## 目录

```text
1-Onnx/
  main.py                              # 服务入口
  app/
    bootstrap/                         # 应用创建、启动关闭、Nacos 配置更新
    core/
      config/
        constant.py                    # 部署时修改：端口、日志、Nacos 等
        settings.py                    # 类型化启动配置
        loader.py                      # YAML 读取、模型路径解析
        nacos.py                       # 配置拉取、本地回退、GET 轮询
      observability/                   # 普通日志、平台 JSON 日志、traceId
    adapter/
      http/                            # 原 /process/ 接口和请求日志
    domain/
      pipeline/
        a_gdinoDtct.py                  # GdinoDetect：配置读取校验、模型初始化、预热和检测
        b_clipdtct.py                   # ClipDetect：CLIP 图文匹配
      common/
        onnx_utils.py                  # 共用 ORT 会话、provider、日志和 CUDA DLL 加载
      groundingdino/
        preprocess.py                  # 图片、文本预处理和输入构造
        postprocess.py                 # 阈值筛选和短语解码
        tokenizer.py                   # 本地 BERT 分词器
        visualize.py                   # 检测框标注
  config/
    param/config.yaml                  # 模型参数；Nacos 使用相同结构
    model/groundingdino/
      groundingdino_swint_ogc.onnx      # 包含 BERT、视觉网络和检测网络
      groundingdino_swint_ogc.metadata.json
      bert-base-uncased/                # 分词器文件，无需额外 BERT 权重
    images/test.jpg                    # 测试图片
  output/
    logs/app.log                       # 普通运行日志
    logs/platform.log                  # 每行一条 JSON 的平台日志
    images/                            # HTTP 上传后的 JPEG 图片
  scripts/
    infer.py                           # 直接推理 CLI
    infer_folder.py                    # 独立单文件：目录批量推理、结果和耗时 JSON
    test_api.py                        # HTTP 调用和结果标注示例
  requirements.txt                     # Linux x86_64 GPU 运行依赖
  tests/
    test_config.py                     # unittest：配置、Nacos 和日志检查
    check_api.py                       # 真实模型 HTTP 集成检查
    fixtures/openapi.json              # 原接口协议基线
  tools/
    onnx_export.py                     # 从原 PyTorch 模型导出
    validate_onnx.py                   # PyTorch/ONNX 检测结果对比
  reports/                             # 检查结果
```

模型约 661 MiB，batch=1，图片宽高动态，文本最多 256 token。模型和分词器位于 `config/model/groundingdino/`，配置文件中的相对路径均以本项目目录为根，与运行命令时所在目录无关。运行时直接使用 `tokenizers` 加载本地 `tokenizer.json`，不需要安装 `transformers`。

服务与单图脚本 `scripts/infer.py` 共用 `domain/pipeline/a_gdinoDtct.py` 的 `GdinoDetect`。它按 `b_clipdtct.py` 的形式组织配置读取、模型初始化、预热和检测，`_load_config()` 直接读取并校验配置字典，将模型参数保存到检测器属性；GDINO 的分词与前后处理保留在 `domain/groundingdino/`，通用 ONNX 运行环境集中在 `domain/common/onnx_utils.py`。HTTP 层负责上传和响应；启动层将配置、模型和上传目录统一保存在 `app.state.runtime`。目录批量脚本 `scripts/infer_folder.py` 的加载、预处理、推理、后处理和结果写入均在同一文件内实现，只依赖标准库及第三方库，不导入本项目其他 Python 文件。

部署时保留 `main.py`、`app/`、`config/` 和所选的依赖文件即可，`output/` 会在运行时生成。`scripts/` 用于人工调用，`tests/` 用于回归，`tools/` 用于模型导出与对比，`reports/` 保存验证结果；这些开发辅助目录不参与服务启动。

## 修改参数

服务启动参数在 **`app/core/config/constant.py`**，修改后重启：

| 参数 | 用途 / 默认值 |
| --- | --- |
| `DEFAULT_HOST` / `DEFAULT_PORT` | 监听地址和端口，`0.0.0.0:8678` |
| `APP_RELOAD` | 开发自动重载，默认关闭 |
| `DEFAULT_YAML_PATH` | 本地参数文件，`config/param/config.yaml` |
| `LOGGER_NAME` / `LOG_LEVEL` | 平台索引名、日志级别 |
| `ORT_LOG_SEVERITY_LEVEL` | ORT 原生日志级别：0 详细、1 信息、2 警告、3 错误、4 致命；默认 3 |
| `LOG_SAVE_TO_FILE` | 保存 `app.log` 和 `platform.log`，默认开启 |
| `LOG_SAVE_K8S_FILE` | 将平台 JSON 同时输出到 stdout，默认关闭 |
| `LOG_BACKUP_COUNT` | 日志按天轮转，保留 15 份历史文件 |
| `LOG_TIMEZONE_OFFSET_HOURS` | 日志与轮转时区，默认 UTC+8，与参考项目一致 |
| `OUTPUT_DIR` | 日志输出根目录，`output` |
| `SAVE_IMAGES_DIR` / `SAVE_IMAGES_ENABLED` | 上传图片目录、是否保留图片 |
| `JPEG_QUALITY` | 上传图片重编码质量，默认 95 |
| `WARMUP_ENABLED` / `WARMUP_RUNS` | 启动预热开关和次数，默认关闭 |
| `NACOS_*` | Nacos 开关、环境、连接信息及轮询间隔 |

模型配置位于 **`config/param/config.yaml`**，与参考项目采用 `TOTAL_CONFIG.device_id`、`model_path`、`use_gpu`、`mean/std` 的方式一致：

```yaml
TOTAL_CONFIG:
  device_id: 0

GROUNDINGDINO_CONFIG:
  model_path: config/model/groundingdino/groundingdino_swint_ogc.onnx
  tokenizer_path: config/model/groundingdino/bert-base-uncased
  use_gpu: true
  threads: 4
  short_side: 800
  max_size: 1333
  max_text_len: 256
  caption: "glass . straw . lemon ."
  test_image_path: config/images/test.jpg
  box_threshold: 0.35
  text_threshold: 0.25
  mean: [0.485, 0.456, 0.406]
  std: [0.229, 0.224, 0.225]
```

`use_gpu: false` 使用 CPU。CUDA 模式会检查实际 provider，初始化失败时启动报错。`threads: 0` 使用 ONNX Runtime 默认线程数。更改缩放、归一化或文本长度参数会改变模型输出；默认值保留已验证的原模型处理方式。

`caption`、`box_threshold`、`text_threshold` 是检测默认值，直接在 YAML 中修改。`GdinoDetect.detect()`、CLI 和服务预热使用 YAML 中的阈值；HTTP 表单与 CLI 显式传入的阈值以调用参数为准。配置中的模型、分词器和测试图片路径均支持绝对路径；相对路径以项目目录为根。

运行 `python -m app.domain.pipeline.a_gdinoDtct` 可直接检测 YAML 中的 `test_image_path`，使用配置中的 `caption` 和阈值。

通过环境变量 `GROUNDING_ONNX_CONFIG` 可指定其他 YAML 文件，适用于服务和单图脚本 `scripts/infer.py`。单图脚本读取本地 YAML；服务启用 Nacos 后，优先使用所选环境的远程配置，获取失败才回退本地 YAML。远程多个 dataId 按列表顺序合并，后面的同名参数覆盖前面的参数。独立批量脚本 `scripts/infer_folder.py` 不读取 YAML 或 `constant.py`。

### Nacos

在 `constant.py` 中填写 `NACOS_CONFIG_PROFILES` 的 `prod` / `test`：`server_addresses`、`namespace`、`username`、`password`、`group`、`data_id_list`，再设置 `NACOS_ENABLED = True`。当前连接参数留空，默认关闭，不会请求参考项目的 Nacos 服务。

通过 `NACOS_DEFAULT_PROFILE` 或环境变量 `NACOS_ENV=prod/test` 选择环境。Nacos 中的 YAML 结构与本地 `config/param/config.yaml` 相同，模型路径指向部署机器的文件。

服务使用普通 GET 轮询，默认每 10 秒检查配置变化。参数变化时先加载替换模型；成功后切换，失败时保留当前模型。远程暂时不可用时保留当前配置并重试。模型变更期间可能同时存在两个会话，需要足够内存/显存。启动时远程不可用会使用本地配置，本次进程不启动远程轮询；连接恢复后重启以重新加载远程配置。

端口、日志和 Nacos 连接本身属于启动参数，修改 `constant.py` 后重启生效。模型 YAML 修改后也需要重启；启用 Nacos 轮询时远程模型参数可自动更新。

## 启动

依赖仅保留一份根目录 `requirements.txt`，用于 Linux x86_64 GPU 部署，使用 Python 3.10、glibc ≥ 2.27：

```bash
python -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

其中锁定 ONNX Runtime 1.19.2、CUDA 12.4 和 cuDNN 9 运行库，兼容关系见 [ORT CUDA 要求](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements)。Linux 主机需要能正常执行 `nvidia-smi` 的 NVIDIA 驱动。

系统需提供 zlib；Ubuntu/Debian 可用 `sudo apt-get install -y zlib1g` 安装。Python 依赖文件同时包含 CUDA Runtime、cuBLAS、cuDNN、cuFFT 和 cuRAND 运行库。

当前 ORT 版本不会自动查找 pip 安装的 NVIDIA `.so` 目录。在启动服务或批量脚本前，使用同一个 Python 环境设置共享库搜索路径：

```bash
NVIDIA_LIB_DIRS=$(python -c 'import site; from pathlib import Path; roots = site.getsitepackages() + [site.getusersitepackages()]; print(":".join(sorted({str(p) for root in roots for p in (Path(root) / "nvidia").glob("*/lib") if p.is_dir()})))')
export LD_LIBRARY_PATH="$NVIDIA_LIB_DIRS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
python main.py
```

在项目根目录运行上述命令。服务统一通过 `python main.py` 或 `uvicorn main:app --host 0.0.0.0 --port 8678` 启动。HTTP 处理代码位于 `app/adapter/http/endpoints/predict.py`。默认接口文档：`http://127.0.0.1:8678/docs`。不需要安装 PyTorch 或完整 CUDA 编译工具链。

## 日志

启动、配置来源、模型路径和 provider、请求参数、检测数量、排队/预处理/推理/后处理耗时、400/422/500 状态和异常堆栈均有日志。普通日志写入 `output/logs/app.log`，结构化日志写入 `output/logs/platform.log`，按配置时区每天轮转。

每个 HTTP 请求生成 `traceId`，贯穿线程池推理日志，同时通过响应头 `X-Trace-Id` 返回；响应 JSON 仍保留原接口的三个字段。平台日志包含 `index/time/env/logger/level/thread/class/message/traceId/stack_trace` 和状态、耗时等业务字段。可以使用与参考项目相同的调用方式：

```python
from app.core.observability.log_helper import show_text

show_text("识别完成", model="groundingdino", detections=10, spendTime=120.5)
```

ORT C++ 日志独立于 Python `LOG_LEVEL`。`ORT_LOG_SEVERITY_LEVEL` 同时控制会话和全局 kernel 日志，默认 3 保留错误并减少启动警告；改为 2 可重新显示警告，排查节点分配时可设为 0。修改后重启生效。该设置只控制日志，不会减少 CPU/GPU 拷贝或改变节点分配。

`Memcpy nodes` 表示图中有设备间数据拷贝，可能影响性能；部分 shape 等辅助节点由 ORT 分配到 CPU，并不表示整个模型在 CPU 上运行。当前模型的两个 `ScatterND` 来自文本分类分数填充到 256 token 的连续切片赋值，索引不重复；该 kernel 的警告是对无 reduction 操作的一般提示。模型和计算图未因日志设置而修改。

## 接口和推理脚本

业务代码可像 `ClipDetect` 一样直接创建检测器，输入 OpenCV 的 BGR 图片和提示词列表：

```python
import cv2

from app.core.config.loader import load_config
from app.domain.pipeline.a_gdinoDtct import GdinoDetect

detector = GdinoDetect(load_config())
detector.warmup(runs=1)
image = cv2.imread("config/images/test.jpg")
result = detector.detect(image, ["glass", "straw", "lemon"])
print(result["boxes"], result["phrases"], result["logits"])
```

提示词也可直接传入字符串，例如 `"glass . straw . lemon ."`。省略时使用配置中的 `caption`；`detect()` 返回归一化 `cxcywh` 框、短语和分数，与 HTTP 的三个结果字段一致。

接口保持 **`POST /process/`**，`multipart/form-data` 必传 `image`、`discri`、`box_thre`、`text_thre`。两个阈值由调用方传入，不会被配置文件替换。

```python
import requests

with open("config/images/test.jpg", "rb") as image:
    response = requests.post(
        "http://127.0.0.1:8678/process/",
        files={"image": ("test.jpg", image, "image/jpeg")},
        data={"discri": "glass . straw . lemon .", "box_thre": "0.35", "text_thre": "0.25"},
        timeout=180,
    )
response.raise_for_status()
print(response.json())
```

成功 JSON 仅包含 `boxes`、`phrases`、`logits`。框格式为归一化 `[中心x, 中心y, 宽, 高]`。无检测时返回三个空列表。非图片和损坏图片返回 400，缺少表单字段/浮点数格式错误返回 422，推理异常返回 500。

HTTP 保留原来的 JPEG quality=95 重编码后再推理流程。默认短边 800、长边最多 1333、ImageNet 归一化，不增加 NMS。`SAVE_IMAGES_ENABLED = False` 会在推理后清理本次 JPEG，编码流程仍执行。

服务启动后另开终端执行：

```powershell
python scripts/test_api.py --output output/client_predictions.json --annotated output/client_result.jpg
```

直接推理脚本自动读取本地 YAML 中的模型、分词器和算法参数：

```powershell
python scripts/infer.py --image config/images/test.jpg --caption "glass . straw . lemon ." --output output/predictions.json --annotated output/result.jpg
```

可以用 `--config` 选择 YAML，用 `--model/--tokenizer/--provider/--device-id/--threads/--short-side/--max-size/--max-text-len` 覆盖本次 CLI 参数。CLI 默认框/文本阈值读取 YAML 的 `box_threshold` 和 `text_threshold`，也可用 `--box-threshold/--text-threshold` 覆盖。CLI JSON 带运行信息，直接推理不做 HTTP 的 JPEG 重编码。

### 目录批量推理

`scripts/infer_folder.py` 一次加载模型，逐张处理目录图片，将所有结果和耗时汇总到一个 UTF-8 JSON 文件。每张图片处理后更新结果快照，坏图会记录错误并继续处理后续图片。该脚本不接收命令行参数，直接修改文件顶部的配置，例如：

```python
INPUT_DIR = Path(r"D:\images")
CAPTION = "glass . straw . lemon ."
OUTPUT_JSON = Path(r"D:\results\batch_predictions.json")
```

然后运行：

```powershell
python scripts/infer_folder.py
```

顶部配置还包括 `MODEL_PATH`、`TOKENIZER_PATH`、`PROVIDER`、`RECURSIVE`、`DEVICE_ID`、`THREADS`、`SHORT_SIDE`、`MAX_SIZE`、`MAX_TEXT_LEN`、`BOX_THRESHOLD`、`TEXT_THRESHOLD`、`ORT_LOG_SEVERITY`。设置 `RECURSIVE = True` 扫描子目录，CPU 环境设置 `PROVIDER = "cpu"`。支持 `.jpg`、`.jpeg`、`.png`、`.bmp`、`.tif`、`.tiff`、`.webp`、`.gif`、`.jfif`。

默认读取项目内的 `config/images/`，使用 `config/model/groundingdino/groundingdino_swint_ogc.onnx` 和 `config/model/groundingdino/bert-base-uncased`，输出 `output/batch_predictions.json`；默认 GPU、提示词 `glass . straw . lemon .`、不递归。这些默认路径以项目目录为根；自行设置的相对路径按当前终端目录解析。所有参数均在该文件内定义，不读取 YAML、`constant.py` 或其他项目 Python 文件。

JSON 的 `summary` 包含 `total_images`、`succeeded`、`failed`、`total_detections`，以及 `model_load_ms`、`processing_time_ms`、`total_inference_ms`、`mean_inference_ms`、`mean_image_time_ms`。每张结果的 `timings_ms` 包含 `preprocess_ms`、`inference_ms`、`postprocess_ms`、`total_ms`：`inference_ms` 只统计 `session.run`，单张总耗时覆盖预处理、模型前向与结果解码，不包含一次性模型加载。

全部图片成功时退出码为 0；任一图片失败时仍写出汇总结果，退出码为 1，模型加载或运行失败也返回 1；配置无效或输入目录没有支持的图片时退出码为 2。

## 验证和重新导出

```powershell
python -m pip install httpx==0.28.1 -i https://pypi.tuna.tsinghua.edu.cn/simple
python -m unittest discover -s tests -p "test_*.py"
python tests/check_api.py --provider cuda
```

配置检查覆盖本地读取、路径解析、参数校验、Nacos 鉴权重试/本地回退/轮询恢复、模型更新失败保留旧模型及日志格式。Nacos 检查使用模拟响应，不访问真实环境。`tests/test_config.py` 也可以单独运行并写入 JSON 报告：

```powershell
python tests/test_config.py
```

接口检查使用真实模型，覆盖原有协议、JPEG 一致性、错误响应和日志，使用 `tests/fixtures/openapi.json` 检查接口定义。需要 GPU 服务依赖；CPU 环境改用 `--provider cpu`。检查不占用服务端口，报告保存到 `reports/`。

本次目录调整前后模型和四个分词器文件的 SHA256、完整 OpenAPI 均相同，见 `reports/structure_assets.json`。原模型导出的一致性范围为原验证样例下的阈值后检测，不表示所有低分候选逐元素一致。

`tools/onnx_export.py` 和 `tools/validate_onnx.py` 默认从相邻的 `../Grounding` 加载原源码、完整 BERT 和 `.pth` 权重。也可用环境变量 `GROUNDING_SOURCE_ROOT` 指定原项目；这些只用于重新导出/校验，不是部署依赖。原 GroundingDINO 的 PyTorch 模型仍依赖 `transformers`，执行这些工具时需要在导出环境中另行安装原项目的依赖：

```powershell
python tools/onnx_export.py
python tools/validate_onnx.py --model config/model/groundingdino/groundingdino_swint_ogc.onnx --provider cuda --device cuda --report reports/revalidated.cuda.json
```

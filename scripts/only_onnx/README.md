# ONNX 推理性能对比包

用来比较不同服务器跑食材检测（`/food_detect/`）的性能，支持 NVIDIA GPU（如 H20，CUDA）和摩尔线程 S5000（MUSA）。整个文件夹可以直接复制到任意服务器上运行：不依赖项目代码、PyTorch、Nacos 或 HTTP 服务。

## 目录

```
only_onnx/
├── requirements.txt       # NVIDIA 用：onnxruntime-gpu 1.19.2 + CUDA 12 / cuDNN 9 运行库
├── requirements-musa.txt  # 摩尔线程用：只补 ORT-MUSA 镜像里缺的 Pillow、tokenizers
├── run.sh                 # Linux 入口：有 pip 装的 NVIDIA 库时设置 LD_LIBRARY_PATH，再运行 benchmark.py
├── benchmark.py           # 在本机跑评测，生成 output/<tag>_<时间>/report.json
├── compare.py             # 汇总多台服务器的 report.json，生成对比报告（Markdown）
├── onnx_pipeline.py       # GroundingDINO + Chinese-CLIP 推理代码（拷自服务，结果与服务逐位一致）
├── cn_tokenizer.py        # CLIP 中文分词器
├── models/                # 约 1.4 GB：GroundingDINO、CLIP 图像/文本 ONNX 模型和分词器
└── images/                # 测试图片，可以再放自己的图片进来
```

## H20 与 S5000 对比步骤

CLIP 默认和 GroundingDINO 用同一个 provider（与服务一致），所以两边都只需要指定 `--provider`。

### H20（NVIDIA）

```bash
python3 -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple   # 每台一次，Python 3.10/3.11，建议用独立 venv/conda
bash run.sh --tag h20
```

需要 NVIDIA 驱动（`nvidia-smi` 能正常运行），不需要安装 CUDA Toolkit。`python3` 不是要用的解释器时：`PYTHON=/opt/conda/envs/onnx/bin/python bash run.sh ...`。

### S5000（摩尔线程，ORT-MUSA 镜像）

在 `onnxruntime:base-v0.28.1-...-musa4.3.8-...` 镜像的容器里运行。如果平台直接给的是这个镜像的容器，把文件夹拷进去即可；如果要自己在宿主机上启动（需要装好 mt-container-toolkit；平台有自己的启动脚本时以它的 GPU 参数为准）：

```bash
IMAGE=mthreads-base-cn-northwest-1.jcr.service.jdcloud.com/inference/onnxruntime:base-v0.28.1-v1-20261028-musa4.3.8-torch2.7.1-py3.10.12-ubuntu22.04-amd64
docker run --rm -it --env MTHREADS_VISIBLE_DEVICES=0 --shm-size 8g \
  -v "$PWD":/workspace/only_onnx -w /workspace/only_onnx "$IMAGE" bash
```

容器里：

```bash
python3 -c "import onnxruntime as o; print(o.__version__, o.get_available_providers())"
# 应输出 1.23.0+musa.e82282d4 ['MUSAExecutionProvider', 'CPUExecutionProvider']
python3 -m pip install -r requirements-musa.txt     # 只装 Pillow、tokenizers
bash run.sh --tag s5000 --provider musa
```

**不要在这个镜像里安装 `requirements.txt`**：其中的 `onnxruntime-gpu` 会覆盖 MUSA 版的 `onnxruntime`。如果已经装了，`pip uninstall onnxruntime-gpu` 后重新安装镜像自带的 MUSA wheel。

如果文件夹以只读方式挂载，用 `--output-dir` 把结果写到别处，例如 `-v /data/results:/results` 加 `--output-dir /results`。

### 对比

把两边的 `output/<tag>_<时间>/` 文件夹拷到同一台机器（任意装了 numpy 的 Python 即可）：

```bash
python3 compare.py output/h20_20261103_101000 output/s5000_20261103_103000
```

第一个报告作为基准。结果会打印出来，并保存为 `output/compare_<时间>.md`。

请求的 GPU provider 没加载成功时（例如 CUDA 库路径不对、在 MUSA 镜像里装了 onnxruntime-gpu），`benchmark.py` 会直接报错退出，不会悄悄退回 CPU，避免拿 CPU 结果和 GPU 结果对比。

## 评测内容

每张图片都会和 `benchmark.py` 中 `CASES` 的每组参数组合成一个用例（默认 2 张图 × 3 组 = 6 个用例）。每个用例先预热 3 次（GPU 初始化、按输入尺寸选择 kernel），再计时 20 次。流程与服务的 `Detector.detect` 完全相同：解码 → GroundingDINO（预处理、推理、后处理 + NMS）→ 对每个框裁剪后跑 CLIP。

计时结束后，会为每个跑在 GPU 上的模型重新建一个开启 ORT profiling 的 session 推理一次，统计有哪些节点被留在了 CPU 上（见下文“算子分配”）。不需要时把 `benchmark.py` 顶部的 `CHECK_PLACEMENT` 改成 `False`。

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--tag` | 主机名 | 服务器名称，用在报告文件夹名和对比表头 |
| `--provider` | cuda | GroundingDINO 的推理设备：`cuda` / `musa` / `cpu` |
| `--clip-provider` | 同 `--provider` | CLIP 的推理设备，默认与服务一致；`cpu` 可以复现修复前服务把 CLIP 跑在 CPU 上的情况 |
| `--device-id` | 0 | GPU 编号 |
| `--runs` / `--warmup` | 20 / 3 | 计时次数 / 预热次数。只用 CPU 时可以改小，例如 `--runs 3` |
| `--output-dir` | `output/` | 报告文件夹的上级目录 |

其他参数（模型路径、图片、阈值、`CASES`、线程数等）在 `benchmark.py` 顶部修改，默认值与 `config/param/config.yaml` 相同。

## 报告内容

- **环境**：CPU、核数、内存、GPU、驱动、显存（`nvidia-smi` 或 `mthreads-gmi -q`），以及 onnxruntime/numpy/opencv/Pillow 版本。如果模型 SHA256 或评测设置和基准不同，会给出提示（CUDA 与 MUSA 都算 GPU，不算设置不同）。
- **耗时汇总**：模型加载时间、首次请求（冷启动）时间，以及各阶段耗时（各用例中位数的平均值）和串行吞吐。括号内是相对基准的加速比，大于 1 表示更快。
- **算子分配**：GPU provider 不支持的算子会被 ORT 留在 CPU 上，并插入 Memcpy 节点在显存和内存之间来回拷贝，这类回退不报错，只会变慢。表中列出 CPU 节点数 / 总节点数、Memcpy 节点数和一次推理里 CPU 节点的耗时。CUDA 本身也会把形状计算类的小算子放在 CPU（本机 RTX 4060 上 GroundingDINO 为 2216 / 6480，约 20 ms），所以基准列列出最多的 8 种 CPU 算子，其他列只列出比基准多出的算子。S5000 比 H20 慢时先看这一节。
- **各用例端到端耗时**：每个用例的中位数和 p90。
- **结果一致性**：每台机器的检测结果都和基准逐框比较，用来确认各机器做的是同样的计算。要求短语相同、IoU ≥ 0.98、分数差 ≤ 0.005。只有一台机器检测到、且分数距离阈值不到 0.005 的框记为“临界”。有不一致时 `compare.py` 返回退出码 1。

`report.json` 中还保存了每个用例的完整统计（median/mean/p90/min/max）、检测结果和完整的算子分配；`vis/` 中是标注图，蓝框是 GroundingDINO 的结果，红框是 CLIP 保留的框。

## 说明

- 服务的 CLIP 以前把 providers 写成 `["CPUExecutionProvider", "CUDAExecutionProvider"]`，CPU 排在前面，整个 CLIP 都跑在 CPU 上。现在已改为和 GroundingDINO 一样 CUDA 优先、使用同一块 GPU。在 RTX 4060 笔记本上，单张裁剪图从 78 ms 降到 6.7 ms。
- 服务代码只认识 `CUDAExecutionProvider`。直接部署到 MUSA 镜像时，GroundingDINO 和 CLIP 都只会打一条警告就退回 CPU。这里测出的 S5000 数据对应的是服务改成使用 `MUSAExecutionProvider` 之后的情况。
- NVIDIA 上所有 CUDA session 都使用 `use_tf32=0`（GroundingDINO 与服务一致），保证是 FP32 计算。MUSA 侧只传了 `device_id`，ORT-MUSA 默认是否使用 TF32 之类的低精度模式未知；如果“结果一致性”里 S5000 的分数差明显偏大，可能是这个原因。
- 两边的 onnxruntime 版本不同（H20 1.19.2，与服务一致；ORT-MUSA 基于 1.23.0），报告的环境表里会列出。
- `mthreads-gmi` 的输出格式没有公开约定，解析不到 GPU 名称时，原始输出保存在 `report.json` 的 `environment.gpu.raw` 中。
- CLIP 的候选文本与 `config.yaml` 一致：`clip_cap` 属于 `CLIP_RECIPE_RANGE`（一道菜、食物）时只和 `CLIP_COMMON_RANGE` 比较，其他情况和食材 + 通用类别比较。`clip_cap` 不在任何范围内时（例如默认用例里的“豆腐”），每个请求会多跑一次 CLIP 文本模型，这部分耗时计入“文本推理”。修改 `config.yaml` 中的范围后，要同步修改 `benchmark.py` 顶部的三个范围。
- `models/*.onnx` 已加入本目录的 `.gitignore`，避免把 1.4 GB 的模型提交到 git。请直接复制整个文件夹（`scp -r` / `rsync`）。

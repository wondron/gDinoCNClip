# ONNX 推理性能对比包

用来比较不同服务器跑食材检测（`/food_detect/`）的性能。整个文件夹可以直接复制到任意服务器上运行：不依赖项目代码、PyTorch、Nacos 或 HTTP 服务。

## 目录

```
only_onnx/
├── requirements.txt     # 只有 ONNX 推理需要的依赖（onnxruntime-gpu 1.19.2 + CUDA 12 / cuDNN 9 运行库）
├── run.sh               # Linux 入口：先设置 LD_LIBRARY_PATH，再运行 benchmark.py
├── benchmark.py         # 在本机跑评测，生成 output/<tag>_<时间>/report.json
├── compare.py           # 汇总多台服务器的 report.json，生成对比报告（Markdown）
├── onnx_pipeline.py     # GroundingDINO + Chinese-CLIP 推理代码（拷自服务，结果与服务逐位一致）
├── cn_tokenizer.py      # CLIP 中文分词器
├── models/              # 约 1.4 GB：GroundingDINO、CLIP 图像/文本 ONNX 模型和分词器
└── images/              # 测试图片，可以再放自己的图片进来
```

## 使用步骤

**1. 安装依赖**（每台服务器一次，Python 3.10/3.11，建议使用独立 venv/conda 环境）

```bash
python3 -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

需要 NVIDIA 驱动（`nvidia-smi` 能正常运行）。不需要安装 CUDA Toolkit。

**2. 在每台服务器上运行评测**

```bash
bash run.sh --tag h20-a          # --tag 是服务器名称，默认使用主机名
```

如果 `python3` 不是要用的解释器：`PYTHON=/opt/conda/envs/onnx/bin/python bash run.sh --tag h20-a`。Windows 上直接运行 `python benchmark.py --tag xxx`。

请求了 CUDA 但没加载成功时会直接报错退出，不会悄悄退回 CPU，避免拿 CPU 结果和 GPU 结果对比。

**3. 把各台机器的 `output/<tag>_<时间>/` 文件夹拷回同一台机器，进行对比**

```bash
python3 compare.py output/h20-a_20261102_231000 output/a10-b_20261102_232500
```

第一个报告作为基准。结果会打印出来，并保存为 `output/compare_<时间>.md`。

## 评测内容

每张图片都会和 `benchmark.py` 中 `CASES` 的每组参数组合成一个用例（默认 2 张图 × 3 组 = 6 个用例）。每个用例先预热 3 次（CUDA 初始化、按输入尺寸选择 kernel），再计时 20 次。流程与服务的 `Detector.detect` 完全相同：解码 → GroundingDINO（预处理、推理、后处理 + NMS）→ 对每个框裁剪后跑 CLIP。

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--runs` / `--warmup` | 20 / 3 | 计时次数 / 预热次数。只用 CPU 时可以改小，例如 `--runs 3` |
| `--provider` | cuda | GroundingDINO 的推理设备 |
| `--clip-provider` | cpu | **服务目前的 CLIP 实际跑在 CPU 上**（见下文），默认按这个情况测；`cuda` 用于测 CLIP 跑在 GPU 上的效果 |
| `--device-id` | 0 | GPU 编号 |

其他参数（模型路径、图片、阈值、`CASES`、线程数等）在 `benchmark.py` 顶部修改，默认值与 `config/param/config.yaml` 相同。

## 报告内容

- **环境**：CPU、核数、内存、GPU、驱动、显存，以及 onnxruntime/numpy/opencv/Pillow 版本。如果模型 SHA256 或评测设置和基准不同，会给出提示。
- **耗时汇总**：模型加载时间、首次请求（冷启动）时间，以及各阶段耗时（各用例中位数的平均值）和串行吞吐。括号内是相对基准的加速比，大于 1 表示更快。
- **各用例端到端耗时**：每个用例的中位数和 p90。
- **结果一致性**：每台机器的检测结果都和基准逐框比较，用来确认各机器做的是同样的计算。要求短语相同、IoU ≥ 0.98、分数差 ≤ 0.005。只有一台机器检测到、且分数距离阈值不到 0.005 的框记为“临界”。有不一致时 `compare.py` 返回退出码 1。

`report.json` 中还保存了每个用例的完整统计（median/mean/p90/min/max）和检测结果；`vis/` 中是标注图，蓝框是 GroundingDINO 的结果，红框是 CLIP 保留的框。

## 说明

- 在 `app/domain/pipeline/b_clipdtct.py` 中，CLIP 的 providers 写成了 `["CPUExecutionProvider", "CUDAExecutionProvider"]`。ORT 按列表顺序分配算子，CPU 排在前面，所以整个 CLIP 都跑在 CPU 上。在 RTX 4060 笔记本上测得单张裁剪图耗时 78 ms，改用 CUDA 后为 6.7 ms。用 `--clip-provider cuda` 可以测出修复后的效果。
- 所有 CUDA session 都使用 `use_tf32=0`（GroundingDINO 与服务一致），这样不同代 GPU 的结果可以直接比较。
- 如果 `clip_cap` 不在 `CLIP_DETECT_RANGE` 中（例如默认用例里的“食物”“豆腐”），服务会对每个框都重新跑一次 CLIP 文本模型。评测保留了这一行为，这部分耗时计入“文本推理”。
- `models/*.onnx` 已加入本目录的 `.gitignore`，避免把 1.4 GB 的模型提交到 git。请直接复制整个文件夹（`scp -r` / `rsync`）。

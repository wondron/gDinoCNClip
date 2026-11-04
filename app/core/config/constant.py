"""部署时修改这里的启动参数；模型参数位于 config/param/config.yaml。

修改后重启服务。默认读取 Nacos，失败时回退本地 YAML；
设置环境变量 NACOS_USE_LOCAL 后直接读取本地 YAML。
"""

from pathlib import Path
from typing import TypedDict

BASE_DIR = Path(__file__).resolve().parents[3]
SERVICE_NAME = "food_detector"
VERSION = "1.0.0"
APP_ENV = "dev"
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8695
APP_RELOAD = True
DEFAULT_YAML_PATH = "config/param/config.yaml"

LOGGER_NAME = "roki_ai_food_detector"
LOG_LEVEL = "INFO"
ORT_LOG_SEVERITY_LEVEL = 3  # ORT 原生日志：0详细、1信息、2警告、3错误、4致命
LOG_SAVE_TO_FILE = True
LOG_SAVE_K8S_FILE = True  # True 时额外向 stdout 输出平台 JSON 日志
LOG_BACKUP_COUNT = 15
LOG_TIMEZONE_OFFSET_HOURS = 8  # 与参考项目 Asia/Shanghai 一致，无需 tzdata
OUTPUT_DIR = "output"
SAVE_RESULT_DIR = "output/save_res"  # 接口 save_result=true 时保存标注结果图
SAVE_IMAGES_DIR = "output/images"  # 兼容旧配置，HTTP 不再保存上传图片
JPEG_QUALITY = 95

DEFAULT_TIME_OUT = 180.0
WARMUP_ENABLED = False
WARMUP_RUNS = 1
WARMUP_IMAGE_PATH = "config/images/test.jpg"
WARMUP_CAPTION = "glass . straw . lemon ."


class NacosProfile(TypedDict):
    profile: str
    server_addresses: str
    namespace: str
    username: str
    password: str
    group: str
    data_id_list: list[str]


NACOS_DEFAULT_PROFILE = "prod"  # NACOS_ENV 未设置或为空用 prod，非空用 test
NACOS_POLL_ENABLED = True
NACOS_CONNECT_TIMEOUT_SECONDS = 2.0
NACOS_REQUEST_TIMEOUT_SECONDS = 3.0
NACOS_POLL_INTERVAL_SECONDS = 10.0
NACOS_RETRY_INTERVAL_SECONDS = 5.0  # 读取失败后的首次重试间隔，连续失败时按指数退避
NACOS_RETRY_MAX_INTERVAL_SECONDS = 60.0  # 指数退避的重试间隔上限
NACOS_CONFIG_PROFILES: tuple[NacosProfile, ...] = (
    {
        "profile": "prod",
        "server_addresses": "http://nacos.myroki.com:8848",
        "namespace": "239acdd2-c154-4d07-ba96-6d52ebc0dccb",
        "username": "roki-ai-prod",
        "password": "IsAbprjbEGj2ME77jxiOR",
        "group": "ROKI_AI_GROUP",
        "data_id_list": ["roki-ai-grounding-dino"],
    },
    {
        "profile": "test",
        "server_addresses": "http://nacos.myroki.com:8848",
        "namespace": "adfc3ec2-a1ea-4b31-9b1a-fafb4228c8ca",
        "username": "roki-ai-test",
        "password": "Q65ILEBfsAmqnPbJqcSig",
        "group": "ROKI_AI_GROUP",
        "data_id_list": ["roki-ai-grounding-dino"],
    },
)

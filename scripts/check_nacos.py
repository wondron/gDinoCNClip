"""检查 Nacos 远程配置能否读取，不启动服务或加载模型。

示例：
    python scripts/check_nacos.py --profile test
    python scripts/check_nacos.py --profile prod --show-config
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path
from time import perf_counter

import requests
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.config.nacos import NacosConfigClient, NacosConfigError
from app.core.config.settings import NacosSettings


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--profile", choices=("prod", "test"),
                        help="指定连接配置；省略时沿用 NACOS_ENV 和 constant.py 的选择")
    parser.add_argument("--show-config", action="store_true",
                        help="显示读取到的完整原始配置；默认仅显示摘要")
    args = parser.parse_args(argv)

    try:
        current = NacosSettings()
        # 诊断始终连接远程；不受服务本地读取开关影响，也不启动轮询。
        cfg = replace(current, profile=args.profile or current.profile,
                      enabled=True, poll_enabled=False)
        profile = next(item for item in cfg.profiles if item["profile"] == cfg.profile)
    except (ValueError, StopIteration) as exc:
        print(f"[FAIL] Nacos 启动配置无效: {exc}", file=sys.stderr)
        return 1

    print(f"profile   : {profile['profile']}")
    print(f"server    : {profile['server_addresses']}")
    print(f"namespace : {profile['namespace'] or '(public)'}")
    print(f"group     : {profile['group']}")
    print(f"dataIds   : {', '.join(profile['data_id_list'])}")
    print(f"timeout   : connect={cfg.connect_timeout_seconds}s, "
          f"read={cfg.request_timeout_seconds}s")
    if not current.enabled:
        print("[NOTE] 已设置 NACOS_USE_LOCAL，服务会读取本地 YAML；本脚本仍检查远程 Nacos。")

    started = perf_counter()
    client = None
    try:
        client = NacosConfigClient(profile, cfg)
        snapshot = client.fetch_snapshot()
    except yaml.YAMLError as exc:
        # YAML 解析异常可能附带配置原文，默认不回显这些内容。
        print(f"[FAIL] Nacos 配置 YAML 解析失败 ({type(exc).__name__})。", file=sys.stderr)
        return 1
    except (NacosConfigError, ValueError) as exc:
        print(f"[FAIL] Nacos 读取或配置校验失败: {exc}", file=sys.stderr)
        return 1
    except requests.RequestException as exc:
        # requests 异常中的 URL 可能含 accessToken，仅输出异常类型。
        print(f"[FAIL] Nacos 网络请求失败 ({type(exc).__name__})。", file=sys.stderr)
        return 1
    finally:
        if client is not None:
            client.close()

    for data_id, text in snapshot.raw_configs.items():
        print(f"[OK] dataId={data_id} | bytes={len(text.encode('utf-8'))}")
    print(f"[OK] 合并后的配置节点: {', '.join(str(key) for key in snapshot.config)}")
    print(f"[SUCCESS] Nacos 远程配置读取及基础结构校验通过，耗时 {perf_counter() - started:.2f}s。")
    if args.show_config:
        for data_id, text in snapshot.raw_configs.items():
            print(f"\n--- dataId: {data_id} ---")
            print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())

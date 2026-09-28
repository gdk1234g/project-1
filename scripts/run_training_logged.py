"""启动训练并实时保存输出，日志独立于训练目录。"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

from src.experiment import PROJECT_ROOT


def stop_child(process: subprocess.Popen | None) -> None:
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="启动基线训练，日志保存到 outputs/training_logs/<name>.log",
        allow_abbrev=False,
    )
    parser.add_argument("--config", type=Path, default=None, help="可选的用户本地 JSON 配置")
    parser.add_argument("--name", required=True, help="唯一实验名称")
    args, passthrough = parser.parse_known_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,99}", args.name):
        parser.error("name 只能包含字母、数字、下划线、连字符和点，且不能以点开头。")
    run_dir = PROJECT_ROOT / "runs" / args.name
    if run_dir.exists():
        parser.error("训练目录已存在，请使用新的 name，以免覆盖已有实验。")

    command = [sys.executable, "-u", "-m", "scripts.train_baseline", "--name", args.name]
    if args.config is not None:
        command.extend(["--config", str(args.config.expanduser().resolve())])
    command.extend(passthrough)
    log_dir = PROJECT_ROOT / "outputs" / "training_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{args.name}.log"
    try:
        log_file = log_path.open("x", encoding="utf-8", buffering=1)
    except FileExistsError:
        parser.error("同名日志已存在，请使用新的 name，以免覆盖旧日志。")
    child_environment = os.environ.copy()
    child_environment["PYTHONIOENCODING"] = "utf-8"
    child_environment["PYTHONUNBUFFERED"] = "1"
    with log_file as log:
        log.write(f"Started UTC: {datetime.now(timezone.utc).isoformat()}\n")
        # A JSON command list preserves argument boundaries on both Windows and Linux.
        log.write("Arguments: " + json.dumps(command, ensure_ascii=False) + "\n\n")
        print(f"训练日志：{log_path}", flush=True)
        process = None
        try:
            process = subprocess.Popen(
                command, cwd=PROJECT_ROOT, env=child_environment,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace", bufsize=1,
            )
            assert process.stdout is not None
            for line in process.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                log.write(line)
            return_code = process.wait()
            log.write(f"\nExit code: {return_code}\n")
            print(f"训练进程退出码：{return_code}；日志：{log_path}", flush=True)
            return return_code
        except KeyboardInterrupt:
            stop_child(process)
            log.write("\nLauncher interrupted; child process terminated.\n")
            return 130
        except BaseException:
            stop_child(process)
            traceback.print_exc(file=log)
            raise
        finally:
            if process is not None and process.stdout is not None:
                process.stdout.close()


if __name__ == "__main__":
    raise SystemExit(main())

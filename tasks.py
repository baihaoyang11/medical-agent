"""跨平台任务入口 —— 不依赖 make，不需要额外依赖。

    python tasks.py verify      # 配置自检 + 语法检查 + 全量单测（不需要 API Key）
    python tasks.py demo-data   # 生成演示数据
    python tasks.py dev         # 启动服务

Windows 上没有 make 也能用；有 make 的环境可以直接 `make verify`，
Makefile 里的每个目标都只是转发到这里，两边行为保证一致。
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable

BOLD, DIM, RED, GREEN, YELLOW, RESET = (
    "\033[1m", "\033[2m", "\033[31m", "\033[32m", "\033[33m", "\033[0m"
)


def _supports_color() -> bool:
    if os.getenv("NO_COLOR"):
        return False
    if sys.platform == "win32":
        # Windows 10+ 终端支持 ANSI，但需要开启 VT 处理
        try:
            import ctypes

            kernel32 = ctypes.windll.kernel32
            kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
            return True
        except Exception:
            return False
    return sys.stdout.isatty()


USE_COLOR = _supports_color()


def c(text: str, color: str) -> str:
    return f"{color}{text}{RESET}" if USE_COLOR else text


def step(title: str) -> None:
    print(f"\n{c('==>', BOLD)} {c(title, BOLD)}")


def run(cmd: list[str], *, allow_fail: bool = False) -> int:
    print(c("$ " + " ".join(cmd), DIM))
    code = subprocess.call(cmd, cwd=str(ROOT))
    if code != 0 and not allow_fail:
        raise SystemExit(code)
    return code


# ────────────────────────────────────────────────────────────────
# 目标
# ────────────────────────────────────────────────────────────────
def setup() -> None:
    step("安装运行依赖")
    run([PY, "-m", "pip", "install", "-r", "requirements.txt"])


def setup_test() -> None:
    step("安装测试依赖（秒级）")
    run([PY, "-m", "pip", "install", "-r", "requirements-test.txt"])


def data() -> None:
    step("检查数据就绪情况")
    run([PY, "scripts/prepare_data.py", "--check"])


def demo_data() -> None:
    step("生成演示数据")
    run([PY, "scripts/prepare_data.py", "--demo"])


def test() -> None:
    step("运行单元测试")
    run([PY, "-m", "pytest", "tests/", "-q"])


def lint() -> None:
    step("语法编译检查")
    code = run(
        [PY, "-m", "compileall", "-q", "core", "agents", "skills",
         "config.py", "agent.py", "model1.py", "vectors1.py"],
        allow_fail=True,
    )
    if code != 0:
        print(c("[FAIL] 存在语法错误", RED))
        raise SystemExit(code)
    print(c("[OK] 语法编译通过", GREEN))


def config_check() -> None:
    """配置自检。注意：没有 Key 时会打印 FAIL，但不阻断 verify ——
    单测本来就不该需要 Key，这里只把问题摆出来。"""
    step("配置自检")
    sys.path.insert(0, str(ROOT))
    try:
        from config import config  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001
        print(c(f"[FAIL] 无法加载 config：{exc}", RED))
        raise SystemExit(1)

    problems = config.validate()
    for item in problems:
        print(c(f"[FAIL] {item}", RED))
    for item in config.warnings():
        print(c(f"[WARN] {item}", YELLOW))

    if problems:
        print(
            c(
                "\n以上问题会导致服务启动后无法正常工作，但不影响单元测试。\n"
                "填好 .env 后重新执行本命令即可。",
                YELLOW,
            )
        )
    else:
        print(c("[OK] 配置校验通过", GREEN))


def verify() -> None:
    step("可复现自检（不需要任何 API Key）")
    config_check()
    lint()
    test()
    print(c("\n[OK] verify 完成：配置检查 + 语法检查 + 单测全部可复现", GREEN))


def dev() -> None:
    step("启动服务 http://localhost:8103/docs")
    run([PY, "agent.py"])


def docker() -> None:
    step("Docker 启动")
    run(["docker", "compose", "up", "--build"])


def clean() -> None:
    step("清理缓存")
    for path in ROOT.rglob("__pycache__"):
        shutil.rmtree(path, ignore_errors=True)
    for path in ROOT.rglob("*.pyc"):
        path.unlink(missing_ok=True)
    shutil.rmtree(ROOT / ".pytest_cache", ignore_errors=True)
    print(c("[OK] 已清理缓存", GREEN))


TARGETS = {
    "setup": setup,
    "setup-test": setup_test,
    "data": data,
    "demo-data": demo_data,
    "test": test,
    "lint": lint,
    "config-check": config_check,
    "verify": verify,
    "dev": dev,
    "docker": docker,
    "clean": clean,
}


def main() -> None:
    # 子进程直接写自己的 stdout，父进程不 flush 会导致输出交替错位
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:  # noqa: BLE001
        pass

    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help", "help"):
        print(c("用法：python tasks.py <目标>", BOLD))
        print()
        for name, fn in TARGETS.items():
            doc = (fn.__doc__ or "").strip().splitlines()
            print(f"  {name:<13}{doc[0] if doc else ''}")
        print()
        print("  verify       配置自检 + 语法检查 + 单测（无需 Key）")
        return

    target = sys.argv[1]
    if target not in TARGETS:
        print(c(f"未知目标：{target}", RED))
        print("可用：" + ", ".join(TARGETS))
        raise SystemExit(2)
    TARGETS[target]()


if __name__ == "__main__":
    main()

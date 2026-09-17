#!/usr/bin/env bash
# -*- coding: utf-8 -*-
#
# cccc 一键启动脚本
# 用途：验证链路、可选启动后端
#
# 用法（请先 conda activate cccc）：
#   ./scripts/quick_start.sh           # 仅验证
#   ./scripts/quick_start.sh --serve  # 验证后启动后端
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

echo "=== cccc 一键启动 ==="
echo "项目根目录: $PROJECT_ROOT"
echo ""

# 在 cccc 环境中执行（conda run 无需当前 shell 已激活）
_run() {
    if command -v conda &>/dev/null && conda env list 2>/dev/null | grep -q "cccc"; then
        conda run -n cccc python "$@"
    else
        python "$@"
    fi
}

# 1. 验证链路
echo ">>> 步骤 1：运行烟雾测试..."
_run -m scripts.run_smoke_tests
SMOKE_EXIT=$?
if [[ $SMOKE_EXIT -ne 0 ]]; then
    echo ""
    echo "[失败] 烟雾测试未通过，请检查环境与依赖。"
    exit $SMOKE_EXIT
fi

echo ""
echo ">>> 烟雾测试全部通过"
echo ""

# 2. 可选启动后端
if [[ "$1" == "--serve" ]]; then
    echo ">>> 步骤 2：启动后端服务..."
    echo "访问 Dashboard: http://localhost:8000/dashboard/"
    echo "访问 API 文档: http://localhost:8000/docs"
    echo "按 Ctrl+C 停止服务"
    echo ""
    if command -v conda &>/dev/null && conda env list | grep -q "cccc"; then
        conda run -n cccc uvicorn backend.main:app --host 0.0.0.0 --port 8000
    else
        uvicorn backend.main:app --host 0.0.0.0 --port 8000
    fi
else
    echo ">>> 启动后端请执行: $0 --serve"
    echo "   或: uvicorn backend.main:app --host 0.0.0.0 --port 8000"
fi

# -*- coding: utf-8 -*-
"""
可选：通过云端 Qwen API 由真人文本派生 AI 样本（rewrite / polish）。
主 MVP 不依赖此脚本。配置环境变量后自行调用，例如：

  export QWEN_API_KEY=...
  export QWEN_API_BASE=https://dashscope.aliyuncs.com/compatible-mode/v1

实现时请使用官方 OpenAI 兼容客户端；本文件仅占位说明，避免误依赖 API Key。
"""

def main() -> None:
    print(
        "占位：未实现网络调用。请在有 API Key 时使用 openai 兼容客户端接入 Qwen，"
        "输出 JSONL 字段 text, label=ai, source=qwen_rewrite|qwen_polish。"
    )


if __name__ == "__main__":
    main()

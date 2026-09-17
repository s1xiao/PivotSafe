# -*- coding: utf-8 -*-
"""
在 hfl/chinese-roberta-wwm-ext 上训练 robust 二分类检测器：冻结 embedding + 前 8 层 encoder，仅训练后 4 层 + 分类头。

数据：JSONL 行对象字段 text, label in {human, ai}。

用法:
  conda run -n cccc python -m scripts.train_robust_detector \\
    --train data/hf_prepared/train.jsonl \\
    --validation data/hf_prepared/val.jsonl \\
    --max-steps 200 --batch-size 8
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROBUST_CFG = PROJECT_ROOT / "configs" / "robust_text_detector.json"


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


class JsonlClsDataset(Dataset):
    def __init__(self, rows: List[Dict[str, Any]], tokenizer, max_len: int) -> None:
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> Dict[str, Any]:
        r = self.rows[i]
        text = str(r["text"])
        lab = str(r["label"]).lower()
        y = 1 if lab == "ai" else 0
        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"][0],
            "attention_mask": enc["attention_mask"][0],
            "labels": torch.tensor(y, dtype=torch.long),
        }


def freeze_bottom_layers(model: Any, n_freeze_encoder_layers: int) -> None:
    backbone = getattr(model, "roberta", None) or getattr(model, "bert", None)
    if backbone is None:
        raise ValueError("不支持的 backbone，需含 roberta 或 bert")
    for p in backbone.embeddings.parameters():
        p.requires_grad = False
    for i in range(n_freeze_encoder_layers):
        for p in backbone.encoder.layer[i].parameters():
            p.requires_grad = False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=Path, required=True)
    ap.add_argument("--validation", type=Path, required=True)
    ap.add_argument("--max-steps", type=int, default=500)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--learning-rate", type=float, default=2e-5)
    args = ap.parse_args()

    with ROBUST_CFG.open("r", encoding="utf-8") as f:
        rcfg = json.load(f)
    backbone = rcfg.get("hf_model_name", "hfl/chinese-roberta-wwm-ext")
    out_rel = rcfg.get("checkpoint_dir", "checkpoints/robust_roberta")
    out_dir = Path(out_rel)
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir
    max_len = int(rcfg.get("max_length", 256))

    train_rows = load_jsonl(args.train)
    val_rows = load_jsonl(args.validation)
    if not train_rows:
        raise SystemExit("train jsonl 为空")

    tokenizer = AutoTokenizer.from_pretrained(backbone)
    model = AutoModelForSequenceClassification.from_pretrained(
        backbone,
        num_labels=2,
        id2label={0: "human", 1: "ai"},
        label2id={"human": 0, "ai": 1},
    )
    freeze_bottom_layers(model, 8)

    ds_tr = JsonlClsDataset(train_rows, tokenizer, max_len)
    ds_va = JsonlClsDataset(val_rows, tokenizer, max_len) if val_rows else None

    use_cuda = torch.cuda.is_available()
    targs = TrainingArguments(
        output_dir=str(out_dir),
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        max_steps=args.max_steps,
        eval_strategy="steps" if ds_va else "no",
        eval_steps=50 if ds_va else None,
        save_steps=max(50, args.max_steps),
        logging_steps=10,
        report_to=[],
        load_best_model_at_end=False,
    )
    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=ds_tr,
        eval_dataset=ds_va,
    )
    trainer.train()
    out_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    print("saved ->", out_dir)


if __name__ == "__main__":
    main()

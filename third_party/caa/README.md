# CAA 代码副本（vendored）

- **来源**：从本机 `CAA/` 目录**复制**的 `tool.py`、`criteria_1.py`、`dataloader_1.py`、`hqaattack_wrapper.py`、`texthacker_wrapper.py`。**请勿修改原始 `CAA` 仓库**；仅在此目录内维护与 cccc 的集成差异。
- **改动摘要（仅副本）**：
  - 包内相对导入（`.tool` / `.criteria_1` / `.dataloader_1`）。
  - `tool.py`：`TensorFlow` 可选导入；新增 `initialize_minimal_cccc` / `initialize_cccc_injected_components` 供 Baseline 检测器注入。
  - `criteria_1.py`：中文词列表走 `jieba` 词性，避免 NLTK 误标。

## 与 CCCC 固定链的关系

`attacks.fixed_chain.run_fixed_chain` 经 `attacks.caa_cccc_bridge` 调用本目录的 **HQAAttack / TextHacker**（Step2/3）；Pivot 在 `attacks/pivot_attack.py`（Anchor + 中文束搜，无 MLM 改写回退）。详见仓库根 `README.md`。

若使用 `CCCC_CAA_SYNONYMS=legacy`，需要与 CAA 原版一致的数据：

1. **词向量**：`counter-fitted-vectors.txt`（每行 `word dim1 dim2 ...`）
2. **邻接表**：`cos_sim_matrix.pkl`（与 CAA `load_cosine_similarity_matrix` 兼容）

默认查找路径（可用 `CCCC_CAA_EMBED`、`CCCC_CAA_COS` 覆盖）：

- `third_party/caa/data/counter-fitted-vectors.txt`
- `third_party/caa/data/cos_sim_matrix.pkl`

（英文 counter-fitting 词表与中文攻击语义不一致，主链路已弃用。）

## 依赖

`nltk`（`criteria_1` 英文路径与停用词）；中文路径主要用 `jieba`。

## 功能测试

```bash
conda activate cccc
cd /path/to/cccc
python -m scripts.test_caa_integration
```

仅跑轻量用例（不重复加载 Baseline）：`CCCC_SKIP_CAA_SLOW=1 python -m scripts.test_caa_integration`。

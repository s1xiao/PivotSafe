import numpy as np, spacy, os, webbrowser, tempfile
from .anchor_text import AnchorText


class WhitespaceTokenizer:
    def __init__(self, vocab): self.vocab = vocab

    def __call__(self, text):
        from spacy.tokens import Doc
        words = text.split(" ")
        return Doc(self.vocab, words=words)


def _make_label_fn(predictor, batch_size=32):
    # predictor: ([str]) -> torch.Tensor[N, C]
    def clf(xs):
        # import torch
        # probs = predictor(xs, batch_size=batch_size).detach().cpu().numpy()
        probs = predictor(xs)
        return np.argmax(probs, axis=1)  # int labels

    return clf


def anchor_word_ranking(
        text_tokens, predictor,
        class_names,
        use_unk,
        threshold, 
        beam_size,
        num_samples, 
        onepass, 
        batch_size,
        anchor_bonus,
        query_budget=100,
        save_html_path=None, open_html=False, show_in_notebook=False
):
    """
    输入:
      - List[str]
      - 受害模型预测函数
    输出:
      - List[float]，即词汇重要性
      - List[(token, score)] 打印
    """
    text = " ".join(text_tokens)

    q_counter = {'n': 0}

    nlp = spacy.blank("en")
    nlp.tokenizer = WhitespaceTokenizer(nlp.vocab)
    explainer = AnchorText(
        nlp, class_names=list(class_names),
        use_unk_distribution=use_unk, mask_string='UNK'
    )

    exp = explainer.explain_instance(
    text,
    _make_label_fn(predictor, batch_size),  # 分类器内部的推理批量
    threshold=threshold,
    beam_size=beam_size,
    delta=0.85, 
    tau=0.9,
    batch_size=batch_size,  
    onepass=onepass,
    q_counter=q_counter,
    query_budget=query_budget 
)

    if show_in_notebook:
        try:
            exp.show_in_notebook()
        except Exception:
            pass

    if save_html_path is not None or open_html:
        path = save_html_path or os.path.join(tempfile.gettempdir(), 'anchor_text.html')
        exp.save_to_file(path)
        if open_html:
            webbrowser.open('file://' + os.path.abspath(path))

    anchor_idx = set(exp.features()) if hasattr(exp, "features") else set()

    words, positions, true_label, sample_fn = explainer.get_sample_fn(
        text, _make_label_fn(predictor, batch_size), onepass=onepass, q_counter=q_counter
    )

    n = len(text_tokens)
    scores = np.zeros(n, dtype=float)

    for i in range(n):
        if i in anchor_idx:
            scores[i] += anchor_bonus

    # === NEW: fuse intermediate tuple stats into token scores ===
    feat_stats = {}
    try:
        if hasattr(exp, "exp_map") and isinstance(exp.exp_map, dict):
            feat_stats = exp.exp_map.get('feature_stats', {}) or {}
    except Exception:
        feat_stats = {}
    
    # 选择一个稳妥的指标（best_pc = max(mean*coverage)）
    alpha = 0.8
    for i in range(n):
        st = feat_stats.get(i) or feat_stats.get(str(i))
        if not st:
            continue
        best_prec = float(st.get('best_prec', 0.0))
        best_cov  = float(st.get('best_cov', 0.0))
        best_pc   = float(st.get('best_pc', 0.0))
        fused = best_pc if alpha == 1.0 else (alpha*best_prec + (1.0-alpha)*best_cov)
    
        # 归一化/截断，避免单词重要性被异常值主导
        fused = max(0.0, min(1.0, fused))
        scores[i] += fused
    
    q_used = q_counter['n']
    final_anchor_indices = list(exp.features()) if hasattr(exp, "features") else []
    return (
        scores.tolist(),
        [(text_tokens[i], float(scores[i])) for i in range(n)],
        q_used,
        final_anchor_indices,
    )


if __name__ == "__main__":
    text_ls = ["i'm", 'convinced', 'i', 'could', 'keep', 'a', 'family', 'of', 'five', 'blind', ',', 'crippled', ',',
               'amish', 'people', 'alive', 'in', 'this', 'situation', 'better', 'than', 'these', 'british', 'soldiers',
               'do', 'at', 'keeping', 'themselves', 'kicking']

    print("=======================")
    print("in_put")
    print(f"text_ls:{text_ls}")

    def _dummy_predictor(xs):
        import numpy as np

        return np.tile([0.5, 0.5], (len(xs), 1))

    import_scores, anchor_pairs, q_used, final_idx = anchor_word_ranking(
        text_ls,
        _dummy_predictor,
        class_names=("negative", "positive"),
        use_unk=True,
        threshold=0.90,
        beam_size=2,
        num_samples=200,
        onepass=False,
        batch_size=32,
        anchor_bonus=1.0,
    )

    print("out_put")
    print(f"import_scores:{import_scores}")
    print(f"anchor_word:{sorted(anchor_pairs, key=lambda x: x[1], reverse=True)}")
    print(f"q_used:{q_used} final_anchor_indices:{final_idx}")

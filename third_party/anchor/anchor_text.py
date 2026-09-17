from . import anchor_base
from . import anchor_explanation
import numpy as np
import json
import os
import string
import sys
from io import open
import numpy as np
import re

_EN_WORD = re.compile(r"[A-Za-z][A-Za-z\-']*$") 

def _filter_and_detok(tokens, scores, tokenizer, english_only=True):
    keep = []
    for t, s in zip(tokens, scores):
        if t.startswith("##"):      
            continue
        w = tokenizer.convert_tokens_to_string([t]).strip()
        if not w:
            continue
        if english_only and not _EN_WORD.fullmatch(w):
            continue
        keep.append((w, float(s)))

    if not keep:
        for t, s in list(zip(tokens, scores))[:50]:
            w = tokenizer.convert_tokens_to_string([t]).strip()
            if w:
                keep.append((w, float(s)))
        if not keep: 
            return ["unk"], np.array([1.0], dtype=float)

    words, vals = zip(*keep)
    return list(words), np.array(vals, dtype=float)

def id_generator(size=15):
    """Helper function to generate random div ids. This is useful for embedding
    HTML into ipython notebooks."""
    chars = list(string.ascii_uppercase + string.digits)
    return ''.join(np.random.choice(chars, size, replace=True))

def exp_normalize(x):
    b = x.max()
    y = np.exp(x - b)
    return y / y.sum()

class TextGenerator(object):
    def __init__(self, url="/root/.cache/huggingface/hub/models--distilbert--distilbert-base-cased/snapshots/6ea81172465e8b0ad3fddeed32b986cdcdcffcf0"):
        from transformers import DistilBertTokenizer, DistilBertForMaskedLM
        import torch
        self.torch = torch
        self.url = url
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if url is None:
            self.bert_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-cased')
            self.bert = DistilBertForMaskedLM.from_pretrained('distilbert-base-cased')
        else:
            self.bert_tokenizer = DistilBertTokenizer.from_pretrained(url, local_files_only=True)
            self.bert = DistilBertForMaskedLM.from_pretrained(url, local_files_only=True)
        self.bert.to(self.device)
        self.bert.eval()
        
    def unmask(self, text_with_mask, topk=500, english_only=True):
        torch = self.torch
        tok = self.bert_tokenizer
        mdl = self.bert
    
        # 关键：开启截断，并用滑动窗口避免把 [MASK] 截没了
        enc = tok(
            text_with_mask,
            return_tensors="pt",
            truncation=True,
            max_length=mdl.config.max_position_embeddings,  # DistilBERT 默认 512
            return_overflowing_tokens=True,  # 产生多个 512 窗口
            stride=64                         # 窗口之间保留一些重叠
        )
        input_ids = enc["input_ids"].to(self.device)
        attn_mask = enc["attention_mask"].to(self.device)
        mask_id = tok.mask_token_id
    
        ret = []
        with torch.no_grad():
            # 逐个窗口处理，寻找窗口内的 [MASK] 位置
            for row in range(input_ids.size(0)):
                ids = input_ids[row:row+1]
                am  = attn_mask[row:row+1]
                mask_pos = (ids[0] == mask_id).nonzero(as_tuple=True)[0]
                if mask_pos.numel() == 0:
                    continue  # 这个窗口没有 [MASK]，跳过
                logits = mdl(input_ids=ids, attention_mask=am).logits  # (1, L, vocab)
    
                for pos in mask_pos.tolist():
                    v, idx = torch.topk(logits[0, pos], topk)
                    toks = tok.convert_ids_to_tokens(idx.tolist())
                    words, vals = _filter_and_detok(toks, v.tolist(), tok, english_only=english_only)
                    ret.append((words, vals))
    
        # 兜底，避免上游拿 ret[0] 崩掉
        if not ret:
            ret = [(["unk"], np.array([1.0], dtype=float))]
        return ret



class SentencePerturber:
    def __init__(self, words, tg, onepass=False):
        self.tg = tg
        self.words = words
        self.cache = {}
        self.mask = self.tg.bert_tokenizer.mask_token
        self.array = np.array(words, '|U80')
        self.onepass = onepass
        self.pr = np.zeros(len(self.words))
        for i in range(len(words)):
            a = self.array.copy()
            a[i] = self.mask
            s = ' '.join(a)
            w, p = self.probs(s)[0]
            self.pr[i] =  min(0.5, dict(zip(w, p)).get(words[i], 0.01))
    def sample(self, data):
        a = self.array.copy()
        masks = np.where(data == 0)[0]
        a[data != 1] = self.mask
    
        if self.onepass:
            s = ' '.join(a)
            rs = self.probs(s) 
    
            if len(rs) == len(masks):
                reps = [np.random.choice(words, p=probs) for (words, probs) in rs]
                a[masks] = np.array(reps, dtype=a.dtype)
            else:
                for i in masks:
                    b = self.array.copy()  
                    b[i] = self.mask
                    s_i = ' '.join(b)
                    words, probs = self.probs(s_i)[0]
                    a[i] = np.random.choice(words, p=probs)
        else:
            for i in masks:
                s = ' '.join(a)
                words, probs = self.probs(s)[0]
                a[i] = np.random.choice(words, p=probs)
    
        return a


    def probs(self, s):
        if s not in self.cache:
            r = self.tg.unmask(s)
            self.cache[s] = [(a, exp_normalize(b)) for a, b in r]
            if not self.onepass:
                self.cache[s] = self.cache[s][:1]
        return self.cache[s]


    def perturb_sentence(self, present, n, prob_change=0.5):
        raw = np.zeros((n, len(self.words)), '|U80')
        data = np.ones((n, len(self.words)))




class AnchorText(object):
    """bla"""
    class QueryBudgetExceeded(Exception):
        pass

    def __init__(self, nlp, class_names, use_unk_distribution=True, mask_string='UNK'):
        """
        Args:
            nlp: spacy object
            class_names: list of strings
            use_unk_distribution: if True, the perturbation distribution
                will just replace words randomly with mask_string.
                If False, words will be replaced by similar words using word
                embeddings
            mask_string: String used to mask tokens if use_unk_distribution is True.
        """
        self.nlp = nlp
        self.class_names = class_names
        self.use_unk_distribution = use_unk_distribution
        self.tg = None
        self.mask_string = mask_string
        if not self.use_unk_distribution:
            self.tg = TextGenerator()

    def get_sample_fn(self, text, classifier_fn, onepass=False, use_proba=False, q_counter=None, query_budget=None):
        # print("NOW IS IN GET")
        true_label = classifier_fn([text])[0]
        # q_counter['n'] += 1  # +=1
        processed = self.nlp(text)
        words = np.array([x.text for x in processed], dtype='|U80')
        positions = [x.idx for x in processed]
        # positions = list(range(len(words)))
        perturber = None
        if not self.use_unk_distribution:
            perturber = SentencePerturber(words, self.tg, onepass=onepass)
        def sample_fn(present, num_samples, compute_labels=True):
            if self.use_unk_distribution:
                data = np.ones((num_samples, len(words)))
                raw = np.zeros((num_samples, len(words)), '|U80')
                raw[:] = words
                for i, t in enumerate(words):
                    if i in present:
                        continue
                    n_changed = np.random.binomial(num_samples, .5)
                    changed = np.random.choice(num_samples, n_changed,
                                               replace=False)
                    raw[changed, i] = self.mask_string
                    data[changed, i] = 0
                raw_data = [' '.join(x) for x in raw]
            else:
                data = np.zeros((num_samples, len(words)))
                for i in range(len(words)):
                    if i in present:
                        continue
                    probs = [1 - perturber.pr[i], perturber.pr[i]]
                    data[:, i] = np.random.choice([0, 1], num_samples, p=probs)
                data[:, present] = 1
                raw_data = []
                for i, d in enumerate(data):
                    r = perturber.sample(d)
                    data[i] = r == words
                    raw_data.append(' '.join(r))
            labels = []
            if compute_labels:
                ### 改成0.8*query_budget
                if q_counter['n'] + num_samples > 0.8 * query_budget:
                    raise AnchorText.QueryBudgetExceeded()
                labels = (classifier_fn(raw_data) == true_label).astype(int)
                q_counter['n'] += len(raw_data)  # +=num_samples len(raw_data)
            labels = np.array(labels)
            # print(f"q_counter_get_sample_fn01::{q_counter['n']}")
            if query_budget is not None and q_counter['n'] > query_budget:
                raise AnchorText.QueryBudgetExceeded()
            max_len = max([len(x) for x in raw_data])
            dtype = '|U%d' % (max(80, max_len))
            raw_data = np.array(raw_data, dtype).reshape(-1, 1)
            return raw_data, data, labels
        return words, positions, true_label, sample_fn

    def explain_instance(self, text, classifier_fn, threshold=0.85,
                          delta=0.1, tau=0.15, batch_size=10, onepass=False,
                          use_proba=False, beam_size=4, q_counter=None, query_budget=None, **kwargs):
        if type(text) == bytes:
            text = text.decode()

        # print("BEFORE GET IN EX")
        words, positions, true_label, sample_fn = self.get_sample_fn(
            text, classifier_fn, onepass=onepass, use_proba=use_proba, q_counter=q_counter, query_budget=query_budget)
        # print("AFTER GET IN EX")
        # print(f"true_label::{true_label}")
        # print(f"sample_fn::{sample_fn}")
        # print(f"q_counter_get_sample_fn::{q_counter}")
        # print words, true_label
        try:
            exp = anchor_base.AnchorBaseBeam.anchor_beam(
                sample_fn, delta=delta, epsilon=tau, batch_size=batch_size,
                desired_confidence=threshold, stop_on_first=True,
                coverage_samples=10000, **kwargs)
        except AnchorText.QueryBudgetExceeded:
            exp = {'feature': [], 'mean': [], 'precision': [],
                   'coverage': [], 'examples': [], 'all_precision': 0}
        exp['names'] = [words[x] for x in exp['feature']]
        exp['positions'] = [positions[x] for x in exp['feature']]
        exp['instance'] = text
        exp['prediction'] = true_label
        explanation = anchor_explanation.AnchorExplanation('text', exp,
                                                           self.as_html)
        explanation.exp_map['query'] = q_counter['n'] #
        # === NEW: forward feature_stats to exp_map if available ===
        if isinstance(exp, dict) and 'feature_stats' in exp:
            explanation.exp_map['feature_stats'] = exp['feature_stats']
        return explanation

    def as_html(self, exp):
        predict_proba = np.zeros(len(self.class_names))
        exp['prediction'] = int(exp['prediction'])
        predict_proba[exp['prediction']] = 1
        predict_proba = list(predict_proba)

        def jsonize(x):
            return json.dumps(x)
        this_dir, _ = os.path.split(__file__)
        bundle = open(os.path.join(this_dir, 'bundle.js'), encoding='utf8').read()
        random_id = 'top_div' + id_generator()

        example_obj = []

        def process_examples(examples, idx):
            idxs = exp['feature'][:idx + 1]
            out_dict = {}
            new_names = {'covered_true': 'coveredTrue', 'covered_false': 'coveredFalse', 'covered': 'covered'}
            for name, new in new_names.items():
                ex = [x[0] for x in examples[name]]
                out = []
                for e in ex:
                     processed = self.nlp(str(e))
                     valid = [i for i in idxs if 0 <= i < len(processed)]
                     raw_indexes = [(processed[i].text, processed[i].idx, exp['prediction']) for i in valid]
                     out.append({'text': e, 'rawIndexes': raw_indexes})     # 仅用于展示
                out_dict[new] = out
            return out_dict

        example_obj = []
        for i, examples in enumerate(exp['examples']):
            example_obj.append(process_examples(examples, i))

        explanation = {'names': exp['names'],
                       'certainties': exp['precision'] if len(exp['precision']) else [exp['all_precision']],
                       'supports': exp['coverage'],
                       'allPrecision': exp['all_precision'],
                       'examples': example_obj}
        processed = self.nlp(exp['instance'])
        raw_indexes = [(processed[i].text, processed[i].idx, exp['prediction'])
                       for i in exp['feature']]
        raw_data = {'text': exp['instance'], 'rawIndexes': raw_indexes}
        jsonize(raw_indexes)

        out = u'''<html>
        <meta http-equiv="content-type" content="text/html; charset=UTF8">
        <head><script>%s </script></head><body>''' % bundle
        out += u'''
        <div id="{random_id}" />
        <script>
            div = d3.select("#{random_id}");
            lime.RenderExplanationFrame(div,{label_names}, {predict_proba},
            {true_class}, {explanation}, {raw_data}, "text", "anchor");
        </script>'''.format(random_id=random_id,
                            label_names=jsonize(self.class_names),
                            predict_proba=jsonize(list(predict_proba)),
                            true_class=jsonize(False),
                            explanation=jsonize(explanation),
                            raw_data=jsonize(raw_data))
        out += u'</body></html>'
        return out

    def show_in_notebook(self, exp, true_class=False, predict_proba_fn=None):
        """Bla"""
        out = self.as_html(exp, true_class, predict_proba_fn)
        from IPython.core.display import display, HTML
        display(HTML(out))

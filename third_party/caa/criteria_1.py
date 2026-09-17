
import nltk
from nltk.corpus import stopwords
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
import logging
import string
import re

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 全局变量，用于缓存已下载的资源
_NLTK_INITIALIZED = False
_STOPWORDS_CACHE = None


def _has_cjk(s: str) -> bool:
    return any("\u4e00" <= ch <= "\u9fff" for ch in s)


def _get_pos_jieba_words(text_ls):
    """与 CCCC 中文 jieba 分词列表对齐的简化词性（供 HQA/TextHacker 选词）。"""
    try:
        import jieba.posseg as pseg
    except ImportError:
        return ["OTHER"] * len(text_ls)
    out = []
    for w in text_ls:
        if not w or not str(w).strip():
            out.append("OTHER")
            continue
        segs = list(pseg.cut(str(w)))
        if not segs:
            out.append("OTHER")
            continue
        flag = segs[0].flag
        if flag.startswith("n"):
            out.append("NOUN")
        elif flag.startswith("v"):
            out.append("VERB")
        elif flag.startswith("a"):
            out.append("ADJ")
        elif flag.startswith("d"):
            out.append("ADV")
        else:
            out.append("OTHER")
    return out


def _initialize_nltk():
    """初始化NLTK资源，自动下载所需数据包"""
    global _NLTK_INITIALIZED

    if _NLTK_INITIALIZED:
        return True

    try:
        # 下载所需的NLTK数据包
        required_packages = [
            ('punkt', 'tokenizers/punkt'),
            ('averaged_perceptron_tagger', 'taggers/averaged_perceptron_tagger'),
            ('stopwords', 'corpora/stopwords')
        ]

        for package_name, path in required_packages:
            try:
                nltk.data.find(path)
                logger.debug(f"NLTK包 {package_name} 已存在")
            except LookupError:
                logger.info(f"正在下载NLTK包: {package_name}")
                nltk.download(package_name, quiet=True)

        _NLTK_INITIALIZED = True
        logger.debug("NLTK初始化完成")
        return True

    except Exception as e:
        logger.error(f"NLTK初始化失败: {e}")
        return False


def get_pos(text_ls, debug=False):
    """
    获取文本的词性标注列表，针对对抗攻击优化
    改进了词性识别的准确性，特别是动词识别

    Args:
        text_ls (list or str): 词语列表或字符串文本
        debug (bool): 是否输出调试信息

    Returns:
        list: 词性标注列表，使用简化标签 ['ADJ', 'NOUN', 'VERB', 'ADV', 'OTHER']
    """
    # 输入验证和预处理
    if not text_ls:
        logger.warning("输入为空")
        return []

    # 如果是字符串，自动分词
    if isinstance(text_ls, str):
        if debug:
            logger.info(f"输入为字符串类型，自动分词: '{text_ls[:30]}...'")
        try:
            if _initialize_nltk():
                text_ls = word_tokenize(text_ls)
            else:
                # 简单分词作为备选
                text_ls = text_ls.split()
        except Exception as e:
            logger.error(f"分词失败: {e}")
            text_ls = text_ls.split()

    # 确保是列表类型
    if not isinstance(text_ls, list):
        logger.error(f"输入应为列表或字符串，实际类型: {type(text_ls)}")
        return []

    # 确保所有元素是字符串
    if not all(isinstance(word, str) for word in text_ls):
        logger.error("列表中包含非字符串元素")
        return _enhanced_fallback_pos_tagging(text_ls, debug=debug)

    # 中文词序列：走 jieba 词性，避免 NLTK 误标
    if _has_cjk("".join(text_ls)):
        if debug:
            logger.info("检测到中文词列表，使用 jieba 词性")
        return _get_pos_jieba_words(text_ls)

    # 初始化检查
    if not _initialize_nltk():
        logger.warning("NLTK不可用，使用增强备用词性标注方案")
        return _enhanced_fallback_pos_tagging(text_ls, debug=debug)

    try:
        # 记录输入样本
        if debug:
            sample_text = ' '.join(text_ls[:5]) + ('...' if len(text_ls) > 5 else '')
            logger.info(f"词性标注输入: {sample_text} (共 {len(text_ls)} 个词)")

        # 使用NLTK进行词性标注
        pos_tags = pos_tag(text_ls)

        # 记录原始标注结果
        if debug:
            sample_tags = ', '.join([f"{word}/{tag}" for word, tag in pos_tags[:5]])
            logger.info(f"原始词性标注: {sample_tags}...")

        # 增强的NLTK标签到简化标签的映射
        pos_mapping = {
            # 名词类
            'NN': 'NOUN',  # 名词，单数
            'NNS': 'NOUN',  # 名词，复数
            'NNP': 'NOUN',  # 专有名词，单数
            'NNPS': 'NOUN',  # 专有名词，复数

            # 动词类 - 扩展识别
            'VB': 'VERB',  # 动词，基本形式
            'VBD': 'VERB',  # 动词，过去式
            'VBG': 'VERB',  # 动词，现在分词
            'VBN': 'VERB',  # 动词，过去分词
            'VBP': 'VERB',  # 动词，非第三人称单数现在时
            'VBZ': 'VERB',  # 动词，第三人称单数现在时
            'MD': 'VERB',  # 情态动词

            # 形容词类
            'JJ': 'ADJ',  # 形容词
            'JJR': 'ADJ',  # 形容词，比较级
            'JJS': 'ADJ',  # 形容词，最高级

            # 副词类
            'RB': 'ADV',  # 副词
            'RBR': 'ADV',  # 副词，比较级
            'RBS': 'ADV',  # 副词，最高级
            'WRB': 'ADV',  # 疑问副词

            # 其他类别
            'CD': 'OTHER',  # 基数词
            'DT': 'OTHER',  # 限定词
            'IN': 'OTHER',  # 介词或从属连词
            'PRP': 'OTHER',  # 人称代词
            'PRP$': 'OTHER',  # 所有格代词
            'CC': 'OTHER',  # 并列连词
            'TO': 'OTHER',  # 不定式标记
            'WP': 'OTHER',  # 疑问代词
            'WDT': 'OTHER',  # 疑问限定词
        }

        # 转换为简化标签，并进行后处理优化
        simplified_pos = []
        for i, (word, tag) in enumerate(pos_tags):
            simplified_tag = pos_mapping.get(tag, 'OTHER')

            # 后处理：修正常见的标注错误
            simplified_tag = _post_process_pos_tag(word, simplified_tag, text_ls, i, debug)
            simplified_pos.append(simplified_tag)

        if debug:
            # 详细调试信息
            pos_counts = {}
            for pos in simplified_pos:
                pos_counts[pos] = pos_counts.get(pos, 0) + 1
            logger.info(f"词性分布: {pos_counts}")

            # 显示有攻击价值的词
            attack_worthy = []
            for i, (word, pos) in enumerate(zip(text_ls, simplified_pos)):
                if pos in ['NOUN', 'VERB', 'ADJ', 'ADV'] and len(word) > 2:
                    attack_worthy.append(f"{word}({pos})")
            logger.info(f"潜在可攻击词: {attack_worthy}")

        return simplified_pos

    except Exception as e:
        logger.error(f"NLTK词性标注失败: {e}")
        return _enhanced_fallback_pos_tagging(text_ls, debug=debug)


def pos_filter(ori_pos, new_pos_list):
    """
    词性兼容过滤（与 TextFooler / LimeAttack 原版一致）：
    同义词与原词词性相同，或 NOUN/VERB 互换时视为可接受。
    """
    same = [
        True if ori_pos == new_pos or (set([ori_pos, new_pos]) <= set(["NOUN", "VERB"]))
        else False
        for new_pos in new_pos_list
    ]
    return same


def _post_process_pos_tag(word, tag, text_ls, index, debug=False):
    """
    后处理词性标注结果，修正常见错误

    Args:
        word (str): 当前词
        tag (str): 当前标注
        text_ls (list): 完整词汇列表
        index (int): 当前词在列表中的位置
        debug (bool): 是否输出调试信息

    Returns:
        str: 修正后的词性标注
    """
    word_lower = word.lower()

    # 规则1: 常见动词的强制识别
    definite_verbs = {
        # 常见动词原形和变形
        'compels', 'dominates', 'harvests', 'experiments', 'contains',
        'requires', 'suggests', 'implies', 'demonstrates', 'indicates',
        'represents', 'presents', 'focuses', 'explores', 'examines',
        'reveals', 'displays', 'expresses', 'reflects', 'creates',
        'produces', 'generates', 'establishes', 'maintains', 'develops',
        'improves', 'enhances', 'increases', 'decreases', 'reduces',
        'transforms', 'converts', 'combines', 'separates', 'connects',
        'impacts', 'affects', 'influences', 'determines', 'controls',

        # 第三人称单数形式
        'goes', 'comes', 'makes', 'takes', 'gives', 'sees', 'knows',
        'thinks', 'says', 'tells', 'asks', 'tries', 'uses', 'works',
        'plays', 'runs', 'walks', 'looks', 'watches', 'listens',
        'speaks', 'talks', 'reads', 'writes', 'buys', 'sells',
        'shows', 'helps', 'needs', 'wants', 'likes', 'loves',
        'believes', 'hopes', 'expects', 'remembers', 'learns',

        # 过去式和过去分词
        'convinced', 'impressed', 'surprised', 'amazed', 'shocked',
        'disappointed', 'satisfied', 'frustrated', 'confused', 'pleased'
    }

    if word_lower in definite_verbs:
        if debug and tag != 'VERB':
            logger.info(f"修正: '{word}' {tag} -> VERB (强制动词规则)")
        return 'VERB'

    # 规则2: 处理缩写
    if word in ["'s", "'re", "'ve", "'ll", "'d", "'m", "n't"]:
        if debug and tag != 'OTHER':
            logger.info(f"修正: '{word}' {tag} -> OTHER (缩写规则)")
        return 'OTHER'

    # 规则3: 动词识别增强 - 基于词缀和上下文
    if tag == 'NOUN':  # 被错误标注为名词的可能动词
        # 检查动词词缀
        verb_endings = ['s', 'es', 'ed', 'ing']
        if any(word_lower.endswith(ending) for ending in verb_endings):
            # 获取词根
            if word_lower.endswith('es'):
                root = word_lower[:-2]
            elif word_lower.endswith(('s', 'd')):
                root = word_lower[:-1]
            elif word_lower.endswith('ing'):
                root = word_lower[:-3]
            else:
                root = word_lower

            # 检查词根是否是常见动词
            common_verb_roots = {
                'compel', 'dominate', 'harvest', 'experiment', 'contain',
                'require', 'suggest', 'imply', 'demonstrate', 'indicate',
                'represent', 'present', 'focus', 'explore', 'examine',
                'reveal', 'display', 'express', 'reflect', 'create',
                'produce', 'generate', 'establish', 'maintain', 'develop'
            }

            if root in common_verb_roots:
                if debug:
                    logger.info(f"修正: '{word}' NOUN -> VERB (词根动词规则)")
                return 'VERB'

    # 规则4: 形容词识别增强
    if tag == 'OTHER' and word_lower.endswith(('ed', 'ing')):
        # 可能是形容词化的分词
        context_suggests_adj = False

        # 检查前后文是否有be动词或名词
        if index > 0 and text_ls[index - 1].lower() in ['is', 'was', 'are', 'were', 'being', 'been']:
            context_suggests_adj = True
        elif index < len(text_ls) - 1 and text_ls[index + 1].lower() in ['and', 'but', 'or']:
            context_suggests_adj = True

        if context_suggests_adj:
            if debug:
                logger.info(f"修正: '{word}' OTHER -> ADJ (分词形容词规则)")
            return 'ADJ'

    # 规则5: 副词识别增强
    if word_lower.endswith('ly') and len(word) > 3 and tag != 'ADV':
        if debug:
            logger.info(f"修正: '{word}' {tag} -> ADV (-ly后缀规则)")
        return 'ADV'

    return tag


def _enhanced_fallback_pos_tagging(text_ls, debug=False):
    """
    增强的备用词性标注方案，大幅扩展词汇库和规则
    当NLTK不可用时使用

    Args:
        text_ls (list): 词语列表
        debug (bool): 是否输出调试信息

    Returns:
        list: 简化的词性标注列表
    """
    # 处理不同类型的输入
    if isinstance(text_ls, str):
        if debug:
            logger.warning("备用标注: 输入为字符串，尝试分词")
        text_ls = text_ls.split()

    if not isinstance(text_ls, list):
        logger.error(f"备用标注: 无法处理的输入类型: {type(text_ls)}")
        return []

    if len(text_ls) == 0:
        return []

    pos_ls = []

    # 大幅扩展的词汇表
    enhanced_adjectives = {
        # 基础形容词
        'good', 'bad', 'great', 'excellent', 'terrible', 'amazing', 'awful',
        'wonderful', 'horrible', 'fantastic', 'boring', 'interesting', 'nice',
        'beautiful', 'ugly', 'smart', 'stupid', 'funny', 'sad', 'happy',
        'angry', 'excited', 'tired', 'hungry', 'thirsty', 'disappointed',
        'pleased', 'satisfied', 'frustrated', 'confused', 'surprised',

        # 描述性形容词
        'big', 'small', 'large', 'tiny', 'huge', 'little', 'old', 'new',
        'young', 'fresh', 'clean', 'dirty', 'fast', 'slow', 'quick',
        'bright', 'dark', 'light', 'heavy', 'soft', 'hard', 'smooth',
        'rough', 'warm', 'cold', 'hot', 'cool', 'dry', 'wet',

        # 评价性形容词
        'important', 'serious', 'special', 'different', 'similar', 'same',
        'various', 'certain', 'possible', 'impossible', 'difficult', 'easy',
        'simple', 'complex', 'obvious', 'clear', 'unclear', 'sure', 'unsure',
        'correct', 'wrong', 'right', 'main', 'major', 'minor',
        'recent', 'current', 'previous', 'next', 'last', 'first', 'final',

        # 分词形容词
        'impressed', 'surprised', 'amazed', 'shocked', 'disappointed',
        'satisfied', 'frustrated', 'confused', 'pleased', 'concerned',
        'worried', 'excited', 'interested', 'bored', 'tired', 'relaxed',
        'stressed', 'annoyed', 'embarrassed', 'proud', 'ashamed', 'guilty',
        'frightened', 'scared', 'nervous', 'confident', 'comfortable',
        'uncomfortable', 'related', 'connected', 'separated', 'isolated',
        'involved', 'engaged', 'committed', 'dedicated', 'determined',
        'motivated', 'inspired', 'encouraged', 'discouraged', 'supported',
        'experienced', 'skilled', 'talented', 'gifted', 'qualified',
        'educated', 'trained', 'prepared', 'organized', 'planned',
        'unexpected', 'anticipated', 'predicted', 'estimated', 'calculated'
    }

    enhanced_adverbs = {
        # 程度副词
        'very', 'really', 'quite', 'extremely', 'completely', 'totally',
        'absolutely', 'perfectly', 'entirely', 'fully', 'hardly', 'barely',
        'almost', 'nearly', 'mostly', 'mainly', 'basically', 'generally',
        'particularly', 'especially', 'specifically', 'exactly', 'precisely',
        'approximately', 'roughly', 'largely', 'significantly', 'considerably',
        'substantially', 'dramatically', 'remarkably', 'notably', 'severely',
        'seriously', 'deeply', 'highly', 'strongly', 'intensely', 'heavily',

        # 时间副词
        'now', 'then', 'today', 'yesterday', 'tomorrow', 'recently',
        'currently', 'previously', 'formerly', 'originally', 'initially',
        'finally', 'eventually', 'immediately', 'instantly', 'suddenly',
        'gradually', 'slowly', 'quickly', 'rapidly', 'frequently', 'rarely',
        'always', 'never', 'sometimes', 'often', 'usually', 'normally',
        'regularly', 'constantly', 'continuously', 'permanently', 'temporarily',

        # 方式副词
        'carefully', 'clearly', 'obviously', 'certainly', 'definitely',
        'probably', 'possibly', 'perhaps', 'maybe', 'likely', 'unlikely',
        'fortunately', 'unfortunately', 'surprisingly', 'interestingly',
        'naturally', 'actually', 'literally', 'virtually', 'essentially',
        'basically', 'fundamentally', 'theoretically', 'practically',
        'technically', 'officially', 'formally', 'informally', 'personally',
        'professionally', 'socially', 'politically', 'economically',
        'physically', 'mentally', 'emotionally', 'psychologically'
    }

    enhanced_verbs = {
        # 基础动词
        'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
        'do', 'does', 'did', 'will', 'would', 'could', 'should', 'might', 'may',
        'can', 'must', 'shall', 'ought',

        # 行为动词
        'go', 'come', 'get', 'make', 'take', 'give', 'see', 'know', 'think',
        'say', 'tell', 'ask', 'try', 'use', 'work', 'play', 'run', 'walk',
        'look', 'watch', 'listen', 'hear', 'speak', 'talk', 'read', 'write',
        'buy', 'sell', 'pay', 'cost', 'spend', 'save', 'lose', 'find',
        'show', 'hide', 'open', 'close', 'start', 'stop', 'begin', 'end',
        'help', 'need', 'want', 'like', 'love', 'hate', 'prefer', 'choose',
        'decide', 'believe', 'hope', 'expect', 'remember', 'forget', 'learn',
        'teach', 'understand', 'explain', 'describe', 'discuss', 'argue',
        'agree', 'disagree', 'accept', 'refuse', 'allow', 'prevent',

        # 第三人称单数形式
        'goes', 'comes', 'gets', 'makes', 'takes', 'gives', 'sees', 'knows',
        'thinks', 'says', 'tells', 'asks', 'tries', 'uses', 'works',
        'plays', 'runs', 'walks', 'looks', 'watches', 'listens', 'hears',
        'speaks', 'talks', 'reads', 'writes', 'buys', 'sells', 'pays',
        'costs', 'spends', 'saves', 'loses', 'finds', 'shows', 'hides',
        'opens', 'closes', 'starts', 'stops', 'begins', 'ends', 'helps',
        'needs', 'wants', 'likes', 'loves', 'hates', 'prefers', 'chooses',
        'decides', 'believes', 'hopes', 'expects', 'remembers', 'forgets',
        'learns', 'teaches', 'understands', 'explains', 'describes',
        'discusses', 'argues', 'agrees', 'disagrees', 'accepts', 'refuses',
        'allows', 'prevents',

        # 学术和正式动词
        'compels', 'dominates', 'harvests', 'experiments', 'contains',
        'requires', 'suggests', 'implies', 'demonstrates', 'indicates',
        'represents', 'presents', 'focuses', 'explores', 'examines',
        'reveals', 'displays', 'expresses', 'reflects', 'creates',
        'produces', 'generates', 'establishes', 'maintains', 'develops',
        'improves', 'enhances', 'increases', 'decreases', 'reduces',
        'transforms', 'converts', 'combines', 'separates', 'connects',
        'impacts', 'affects', 'influences', 'determines', 'controls',
        'manages', 'operates', 'functions', 'performs', 'achieves',
        'accomplishes', 'realizes', 'recognizes', 'identifies', 'analyzes',
        'evaluates', 'assesses', 'measures', 'calculates', 'estimates',
        'predicts', 'anticipates', 'expects', 'assumes', 'concludes',
        'recommends', 'suggests', 'proposes', 'advises', 'warns', 'informs',
        'notifies', 'announces', 'declares', 'states', 'claims', 'argues',
        'maintains', 'insists', 'emphasizes', 'stresses', 'highlights',
        'illustrates', 'demonstrates', 'proves', 'confirms', 'verifies',
        'validates', 'supports', 'contradicts', 'opposes', 'challenges'
    }

    # 扩展的词缀规则
    enhanced_noun_suffixes = {
        'tion', 'sion', 'ment', 'ness', 'ity', 'ty', 'ance', 'ence',
        'dom', 'hood', 'ship', 'age', 'ure', 'ism', 'ist', 'er', 'or',
        'ing', 'ful', 'less', 'ward', 'wise', 'like', 'ware', 'work'
    }

    enhanced_adj_suffixes = {
        'able', 'ible', 'al', 'ful', 'ic', 'ical', 'ish', 'ive', 'ous',
        'eous', 'ious', 'less', 'like', 'ward', 'wise', 'ed', 'ing',
        'ary', 'ory', 'ent', 'ant', 'ary', 'ory'
    }

    enhanced_adv_suffixes = {
        'ly', 'ward', 'wise', 'ways', 'long', 'most', 'like'
    }

    enhanced_verb_suffixes = {
        'ize', 'ise', 'fy', 'ify', 'ate', 'en', 'er'
    }

    for i, word in enumerate(text_ls):
        if not isinstance(word, str):
            word = str(word)

        word_lower = word.lower()
        word_clean = word_lower.strip(string.punctuation)

        # 跳过空字符串和单纯标点符号
        if not word_clean or word_clean in string.punctuation:
            pos_ls.append('OTHER')
            continue

        # 处理缩写
        if word in ["'s", "'re", "'ve", "'ll", "'d", "'m", "n't"]:
            pos_ls.append('OTHER')
            continue

        # 基于扩展词汇表的判断（优先级最高）
        if word_clean in enhanced_adjectives:
            pos_ls.append('ADJ')
        elif word_clean in enhanced_adverbs:
            pos_ls.append('ADV')
        elif word_clean in enhanced_verbs:
            pos_ls.append('VERB')

        # 基于增强词缀的判断
        elif word_clean.endswith('ly') and len(word_clean) > 3:
            # 检查是否为形容词+ly构成的副词
            base_word = word_clean[:-2]
            if base_word in enhanced_adjectives or any(base_word.endswith(suf) for suf in enhanced_adj_suffixes):
                pos_ls.append('ADV')
            else:
                pos_ls.append('ADV')  # 大多数-ly结尾的都是副词

        elif word_clean.endswith(('ing', 'ed')) and len(word_clean) > 4:
            # 需要区分动词、形容词和名词
            base_word = word_clean[:-3] if word_clean.endswith('ing') else word_clean[:-2]

            # 检查上下文和基础词汇
            if base_word in enhanced_verbs:
                pos_ls.append('VERB')
            elif i > 0 and text_ls[i - 1].lower() in ['is', 'was', 'are', 'were', 'being', 'been', 'very', 'quite',
                                                      'really']:
                # 前面有be动词或程度副词，更可能是形容词
                pos_ls.append('ADJ')
            elif word_clean.endswith('ing') and i < len(text_ls) - 1 and text_ls[i + 1].lower() in ['of', 'in', 'on',
                                                                                                    'at', 'for',
                                                                                                    'with']:
                # ing形式后跟介词，可能是名词
                pos_ls.append('NOUN')
            else:
                # 默认判断为动词
                pos_ls.append('VERB')

        elif any(word_clean.endswith(suffix) for suffix in enhanced_verb_suffixes):
            pos_ls.append('VERB')
        elif any(word_clean.endswith(suffix) for suffix in enhanced_adj_suffixes):
            pos_ls.append('ADJ')
        elif any(word_clean.endswith(suffix) for suffix in enhanced_adv_suffixes):
            pos_ls.append('ADV')
        elif any(word_clean.endswith(suffix) for suffix in enhanced_noun_suffixes):
            pos_ls.append('NOUN')

        # 基于首字母大写的判断
        elif word[0].isupper() and not word.isupper():
            pos_ls.append('NOUN')  # 专有名词

        # 数字
        elif word_clean.isdigit():
            pos_ls.append('OTHER')

        # 检查第三人称单数动词形式
        elif word_clean.endswith('s') and len(word_clean) > 3:
            base_word = word_clean[:-1]
            if base_word in enhanced_verbs:
                pos_ls.append('VERB')
            else:
                # 可能是复数名词
                pos_ls.append('NOUN')

        # 默认判断
        else:
            # 根据词长和常见模式判断
            if len(word_clean) <= 2:
                pos_ls.append('OTHER')  # 短词通常是代词、介词等
            else:
                pos_ls.append('NOUN')  # 默认为名词

    if debug:
        pos_counts = {}
        for pos in pos_ls:
            pos_counts[pos] = pos_counts.get(pos, 0) + 1
        logger.info(f"增强备用标注词性分布: {pos_counts}")

    return pos_ls


def is_meaningful_word(word, min_length=2):
    """
    判断词汇是否对攻击有意义
    增强了过滤规则

    Args:
        word (str): 待判断的词
        min_length (int): 最小词长要求

    Returns:
        bool: 是否有意义
    """
    if len(word) <= min_length:
        return False

    # 过滤标点符号
    if word in string.punctuation:
        return False

    # 过滤纯数字
    if word.isdigit():
        return False

    # 过滤缩写
    if word in ["'s", "'re", "'ve", "'ll", "'d", "'m", "n't"]:
        return False

    # 过滤常见停用词（即使不在stopwords列表中）
    common_stops = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'as', 'is', 'are', 'was', 'were', 'be',
        'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
        'would', 'could', 'should', 'may', 'might', 'must', 'can', 'shall',
        'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it',
        'we', 'they', 'me', 'him', 'her', 'us', 'them', 'my', 'your',
        'his', 'her', 'its', 'our', 'their'
    }

    if word.lower() in common_stops:
        return False

    return True


def debug_word_filtering(text_ls, word2idx=None, debug=True):
    """
    调试词汇过滤过程，找出具体哪一步过滤了词汇
    增强了调试信息的详细程度

    Args:
        text_ls (list): 词汇列表
        word2idx (dict): 词汇到索引的映射
        debug (bool): 是否输出调试信息
    """
    if not debug:
        return

    # print(f"\n=== 词汇过滤调试 ===")
    # print(f"原始文本: {' '.join(text_ls)}")
    # print(f"词汇总数: {len(text_ls)}")

    # 步骤1：词性分析
    pos_ls = get_pos(text_ls, debug=False)
    # print(f"词性分析结果: {list(zip(text_ls, pos_ls))}")

    # 步骤2：按词性筛选
    pos_pref = ["ADJ", "ADV", "VERB", "NOUN"]
    pos_filtered = []
    for pos in pos_pref:
        for i in range(len(pos_ls)):
            if pos_ls[i] == pos:
                pos_filtered.append((i, text_ls[i], pos_ls[i]))

    # print(f"词性筛选后 ({len(pos_filtered)} 个): {[(word, pos) for _, word, pos in pos_filtered]}")

    # 步骤3：长度筛选
    length_filtered = [(i, word, pos) for i, word, pos in pos_filtered if len(word) > 2]
    # print(f"长度筛选后 (>2字符, {len(length_filtered)} 个): {[(word, pos) for _, word, pos in length_filtered]}")

    # 步骤4：有意义词汇筛选
    meaningful_filtered = [(i, word, pos) for i, word, pos in length_filtered if is_meaningful_word(word, 2)]
    # print(f"有意义筛选后 ({len(meaningful_filtered)} 个): {[(word, pos) for _, word, pos in meaningful_filtered]}")

    # 步骤5：词典检查
    if word2idx is not None:

        in_vocab = []
        not_in_vocab = []
        for i, word, pos in meaningful_filtered:
            if word in word2idx:
                in_vocab.append((i, word, pos))
            else:
                not_in_vocab.append((word, pos))


        # 尝试小写匹配
        case_sensitive_missing = []
        for word, pos in not_in_vocab:
            if word.lower() in word2idx:
                case_sensitive_missing.append((word, word.lower()))

        if case_sensitive_missing:
            print(f"大小写问题 ({len(case_sensitive_missing)} 个): {case_sensitive_missing}")

        # 检查一些特定的遗漏词汇
        specific_checks = ['compels', 'dominates', 'harvests', 'experiments']
        for check_word in specific_checks:
            if check_word in [word for word, _ in not_in_vocab]:
                print(f"⚠️ 特别注意: 动词 '{check_word}' 未被识别!")




def get_attackable_words(text_ls, pos_ls=None, word2idx=None, min_length=2, debug=False):
    """
    获取可用于攻击的词汇列表
    增强了词汇识别能力，特别是动词的识别

    Args:
        text_ls (list): 词汇列表
        pos_ls (list): 词性列表，如果为None则自动获取
        word2idx (dict): 词汇到索引的映射，用于检查词汇是否在词典中
        min_length (int): 最小词长要求
        debug (bool): 是否输出调试信息

    Returns:
        list: 可攻击的词汇列表 [(index, word), ...]
    """
    if debug:
        debug_word_filtering(text_ls, word2idx, debug=True)

    if pos_ls is None:
        pos_ls = get_pos(text_ls, debug=debug)

    attackable_words = []
    pos_pref = ["ADJ", "ADV", "VERB", "NOUN"]

    for pos in pos_pref:
        for i in range(len(pos_ls)):
            if i >= len(text_ls):  # 防止索引越界
                break

            word = text_ls[i]
            word_pos = pos_ls[i]

            # 基本条件检查
            if word_pos != pos:
                continue

            if not is_meaningful_word(word, min_length):
                if debug:
                    logger.debug(f"词汇 '{word}' 不符合有意义标准，跳过")
                continue

            # 检查是否在词典中（先尝试原形，再尝试小写）
            word_in_vocab = False
            vocab_word = word

            if word2idx is not None:
                if word in word2idx:
                    word_in_vocab = True
                elif word.lower() in word2idx:
                    word_in_vocab = True
                    vocab_word = word.lower()
                    if debug:
                        logger.info(f"词汇 '{word}' 使用小写形式 '{vocab_word}' 匹配")
                else:
                    if debug:
                        logger.debug(f"词汇 '{word}' 不在词典中，跳过")
                    continue
            else:
                word_in_vocab = True  # 如果没有词典，不进行此检查

            if word_in_vocab:
                attackable_words.append((i, vocab_word))

    if debug:
        logger.info(f"最终找到 {len(attackable_words)} 个可攻击的词: {[word for _, word in attackable_words[:10]]}")

    return attackable_words


def get_stopwords(language='english'):
    """
    获取停用词集合

    Args:
        language (str): 语言代码，默认为 'english'

    Returns:
        set: 停用词集合
    """
    global _STOPWORDS_CACHE

    # 使用缓存避免重复加载
    if _STOPWORDS_CACHE is not None and language == 'english':
        return _STOPWORDS_CACHE

    if not _initialize_nltk():
        logger.warning("NLTK不可用，使用备用停用词列表")
        return _fallback_stopwords()

    try:
        # 使用NLTK获取停用词
        stop_words = set(stopwords.words(language))

        # 添加一些额外的常用停用词
        additional_stopwords = {
            "'s", "'re", "'ve", "'ll", "'d", "'m", "n't",  # 缩写
            "could", "would", "should", "might", "must",  # 情态动词
            "said", "say", "says", "saying",  # 言论动词
            "one", "two", "first", "second", "last",  # 数词和序数词
            "also", "well", "now", "just", "really"  # 常用副词
        }

        stop_words.update(additional_stopwords)

        # 缓存结果（仅限英语）
        if language == 'english':
            _STOPWORDS_CACHE = stop_words

        logger.debug(f"成功加载{language}停用词，共{len(stop_words)}个")
        return stop_words

    except Exception as e:
        logger.error(f"加载NLTK停用词失败: {e}")
        return _fallback_stopwords()


def _fallback_stopwords():
    """备用停用词列表"""
    fallback_words = {
        # 人称代词
        'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves',
        'you', "you're", "you've", "you'll", "you'd", 'your', 'yours',
        'yourself', 'yourselves', 'he', 'him', 'his', 'himself',
        'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its',
        'itself', 'they', 'them', 'their', 'theirs', 'themselves',

        # 疑问词
        'what', 'which', 'who', 'whom', 'this', 'that', "that'll",
        'these', 'those',

        # be动词
        'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being',

        # 助动词
        'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing',
        'will', 'would', 'should', 'could', 'can', 'may', 'might', 'must',

        # 冠词
        'a', 'an', 'the',

        # 连词和介词
        'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while',
        'of', 'at', 'by', 'for', 'with', 'through', 'during', 'before',
        'after', 'above', 'below', 'up', 'down', 'in', 'out', 'on', 'off',
        'over', 'under', 'again', 'further', 'then', 'once',

        # 其他常用词
        'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any',
        'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such',
        'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too',
        'very', 's', 't', 'just', 'don', "don't", 'now', 'd', 'll', 'm',
        'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't",
        'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn',
        "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn',
        "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan',
        "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren',
        "weren't", 'won', "won't", 'wouldn', "wouldn't"
    }

    logger.debug(f"使用备用停用词列表，共{len(fallback_words)}个")
    return fallback_words


def is_stopword(word, stopwords_set=None):
    """检查单词是否为停用词"""
    if stopwords_set is None:
        stopwords_set = get_stopwords()
    return word.lower() in stopwords_set


def filter_stopwords(text_ls, keep_pos=None):
    """过滤文本中的停用词"""
    stopwords_set = get_stopwords()

    if keep_pos is not None:
        pos_ls = get_pos(text_ls)

    filtered_words = []
    filtered_indices = []

    for i, word in enumerate(text_ls):
        is_stop = is_stopword(word, stopwords_set)

        # 如果指定了词性过滤，检查词性
        if keep_pos is not None:
            word_pos = pos_ls[i]
            keep_by_pos = word_pos in keep_pos
        else:
            keep_by_pos = True

        # 保留非停用词且符合词性要求的词
        if not is_stop and keep_by_pos:
            filtered_words.append(word)
            filtered_indices.append(i)

    return filtered_words, filtered_indices



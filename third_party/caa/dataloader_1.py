def read_corpus(file_path, max_lines=None, csvf=False):
    """
    读取MR格式的语料库文件，返回分词后的文本

    参数:
        file_path (str): 文件路径
        max_lines (int): 最大读取行数（可选）
        csvf (bool): 是否为CSV格式（本函数中未使用，为兼容保留）

    返回:
        texts (list): 分词后的文本列表，每个元素是词汇列表
        labels (list): 标签列表（数值类型）
    """
    texts = []
    labels = []

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            # 如果指定了最大行数，只读取前 max_lines 行
            if max_lines is not None:
                lines = []
                for i, line in enumerate(f):
                    if i >= max_lines:
                        break
                    lines.append(line.strip())
            else:
                lines = [line.strip() for line in f.readlines()]

            for line in lines:
                if not line:  # 跳过空行
                    continue

                # 分割标签和文本（第一个空格作为分隔符）
                parts = line.split(' ', 1)

                if len(parts) < 2:
                    continue  # 跳过格式不正确的行

                label_str = parts[0]
                text = parts[1].strip()

                # 将标签转换为数值类型
                try:
                    # 尝试直接转换为整数
                    label = int(label_str)
                except ValueError:
                    # 如果转换失败，尝试识别常见标签格式
                    if label_str.lower() in ["positive", "pos", "1"]:
                        label = 1
                    elif label_str.lower() in ["negative", "neg", "0"]:
                        label = 0
                    else:
                        # 如果无法识别，使用默认值0并记录警告
                        print(f"警告: 无法识别的标签 '{label_str}'，将视为0")
                        label = 0

                # **关键修改：对文本进行分词**
                text_words = text.split()  # 简单的空格分词

                # 可选：更高级的分词处理
                # 清理和标准化
                cleaned_words = []
                for word in text_words:
                    # 移除前后的标点符号但保留缩写
                    word = word.strip('.,!?;:"()[]{}')
                    if word:  # 确保不是空字符串
                        cleaned_words.append(word.lower())  # 转为小写

                # 添加到结果列表
                labels.append(label)
                texts.append(cleaned_words)  # 添加词汇列表而不是字符串

    except Exception as e:
        print(f"读取文件时出错: {e}")
        return [], []

    return texts, labels
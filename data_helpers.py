import numpy as np
import re
import itertools
from collections import Counter
import jieba


def word_seg(text):
    text = text.replace('\u3000', ' ')
    text = ' '.join(jieba.cut(text, HMM=False))
    text = re.sub(r"\s{2,}", " ", text)
    return text


def get_classes():
    class_list = ['体育', '健康', '军事', '娱乐', '教育', '文化', '时尚', '科技', '财经', '汽车']
    cls_to_id = dict((c, i) for i, c in enumerate(class_list))
    id_to_cls = dict((i, c) for i, c in enumerate(class_list))
    return class_list, cls_to_id, id_to_cls


def load_data_and_labels(file_name):
    with open(file_name, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        lines = [line.split('\t') for line in lines]
        x_text = [word_seg(line[3]) for line in lines]
        y_text = [line[0] for line in lines]
        [_, cls_2_id, _] = get_classes()
        y = [cls_2_id[item] for item in y_text]
        y = np.array(y)
        y_key = np.zeros((y.size, y.max() + 1))
        y_key[np.arange(y.size), y] = 1
        return [x_text, y_key]


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]

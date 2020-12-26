import pandas as pd
import six
import numpy as np
from tqdm import tqdm
import traceback
import jieba
import codecs
from .text_utils import WordCounter


def cut(text, type='word'):
    return list(jieba.cut(text))


def filter_duplicate_space(text):
    return ''.join([x for i, x in enumerate(text) if not (i < len(text) - 1 and not(x.strip()) and not(text[i + 1].strip()))])


def filter_quoto(text):
    return text.replace("''", '" ').replace("``", '" ').replace("\n","").replace("\r","")


def remove_duplicate(text):
    if six.PY2:
        text = text.decode("utf8")
    l = []
    start, end = 0, 0
    duplicate = []
    while start < len(text):
        while end < len(text) and text[start] == text[end]:
            duplicate.append(text[start])
            end += 1
        l.append(''.join(duplicate[:5]))  # 为啥是5
        duplicate = []
        start = end
    text = ''.join(l)
    if six.PY2:
        text = text.encode('utf8')
    return text


def text_filter(x):
    x = x.strip('"')
    x = filter_duplicate_space(x)
    x = remove_duplicate(x)
    # x = x.lower()
    return x


def segment_char(text, cn_only=False):
    if not six.PY2 and not cn_only:
        return [x.strip() for x in text if x.strip()]
    l = []
    pre_is_cn = False
    if six.PY2:
        unicode_text = text.decode('utf-8', 'ignore')
    else:
        unicode_text = text
    for word in unicode_text:
        #print('-----------word', word, pre_is_cn)
        if u'\u4e00' <= word <= u'\u9fff':
            pre_is_cn = True
            if l:
                l.append(' ')
        else:
            l.append(' ')
            if pre_is_cn:
                pre_is_cn = False
        if not cn_only or pre_is_cn:
            l.append(word)
    text = ''.join(l)
    if six.PY2:
        text = text.encode('utf-8')
    l = text.split()
    return [x.strip() for x in l if x.strip()]


def gen_content(files):
    contents = []
    for csvfile in files:
        df = pd.read_csv(csvfile)
        for row in tqdm(df.iterrows()):
            contents.append(text_filter(row[1][1]))
    return contents

def segment_basic_single_all(text):
    #results = [word for word in get_single_cns(text)]
    results = [word for word in segment_char(text, cn_only=False)]
    results += [word for word in cut(text)]
    return results


def gen_vocab(contents, vocab_file):
    min_count = 1
    most_common = 0
    counter = WordCounter(most_common=most_common, min_count=min_count)

    START_WORD = '<S>'
    END_WORD = '</S>'

    for i, line in tqdm(enumerate(contents)):
        text = line.rstrip()
        text = text_filter(text)
        try:
            words = segment_basic_single_all(text)
        except Exception:
            print(i, '-----------fail', text)
            print(traceback.format_exc())
            continue
        counter.add(START_WORD)
        for word in words:
            counter.add(word)
            if word.isdigit():
                counter.add('<NUM>')
        counter.add(END_WORD)
    counter.add(START_WORD)
    # counter.save(FLAGS.out_dir + '/%s.txt' % vocab_name)
    counter.save(vocab_file)
    return True


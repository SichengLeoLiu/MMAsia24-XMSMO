""" utility functions"""
import re
import os
from os.path import basename

import gensim
import torch
from torch import nn
import json
from statistics import median

PAD = 0
UNK = 1
START = 2
END = 3
PAD_TOKEN = '<pad>'
UNK_TOKEN = '<unk>'
START_TOKEN = '<start>'
END_TOKEN = '<end>'

import torch
import numpy as np
from torch.autograd import Variable
from collections import defaultdict, Counter, OrderedDict
from os.path import join
from itertools import chain


class OrderedCounter(Counter, OrderedDict):
    """Counter that remembers the order elements are first encountered"""
    def __repr__(self):
        return '%s(%r)' % (self.__class__.__name__, OrderedDict(self))

    def __reduce__(self):
        return self.__class__, (OrderedDict(self),)


def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return x


def idx2word(idx, i2w, pad_idx):
    sent_str = [str()]*len(idx)
    for i, sent in enumerate(idx):
        for word_id in sent:
            if word_id == pad_idx:
                break
            sent_str[i] += i2w[str(word_id.item())] + " "
        sent_str[i] = sent_str[i].strip()
    return sent_str


def interpolate(start, end, steps):

    interpolation = np.zeros((start.shape[0], steps + 2))

    for dim, (s, e) in enumerate(zip(start, end)):
        interpolation[dim] = np.linspace(s, e, steps+2)

    return interpolation.T


def expierment_name(args, ts):
    exp_name = str()
    exp_name += "BS=%i_" % args.batch_size
    exp_name += "LR={}_".format(args.learning_rate)
    exp_name += "EB=%i_" % args.embedding_size
    exp_name += "%s_" % args.rnn_type.upper()
    exp_name += "HS=%i_" % args.hidden_size
    exp_name += "L=%i_" % args.num_layers
    exp_name += "BI=%i_" % args.bidirectional
    exp_name += "LS=%i_" % args.latent_size
    exp_name += "WD={}_".format(args.word_dropout)
    exp_name += "ANN=%s_" % args.anneal_function.upper()
    exp_name += "K={}_".format(args.k)
    exp_name += "X0=%i_" % args.x0
    exp_name += "TS=%s" % ts

    return exp_name
    
def count_data(path):
    """ count number of data in the given path"""
    matcher = re.compile(r'[0-9]+\.json')
    match = lambda name: bool(matcher.match(name))
    names = os.listdir(path)
    n_data = len(list(filter(match, names)))
    return n_data

def make_vocab(wc, vocab_size):
    word2id, id2word = {}, {}
    word2id[PAD_TOKEN] = PAD
    word2id[UNK_TOKEN] = UNK
    word2id[START_TOKEN] = START
    word2id[END_TOKEN] = END
    for i, (w, _) in enumerate(wc.most_common(vocab_size), 4):
        word2id[w] = i
    return word2id

def convert_word2id(w, word2id):
    try:
        wid = word2id[w]
        if wid < 30000:
            return wid
        return UNK
    except:
        return UNK

def make_embedding(id2word, w2v_file, initializer=None):
    attrs = basename(w2v_file).split('.')  #word2vec.{dim}d.{vsize}k.bin
    w2v = gensim.models.Word2Vec.load(w2v_file).wv
    vocab_size = len(id2word)
    emb_dim = int(attrs[-3][:-1])
    embedding = nn.Embedding(vocab_size, emb_dim).weight
    if initializer is not None:
        initializer(embedding)

    oovs = []
    with torch.no_grad():
        for i in range(len(id2word)):
            # NOTE: id2word can be list or dict
            if i == START:
                embedding[i, :] = torch.Tensor(w2v['<s>'])
            elif i == END:
                embedding[i, :] = torch.Tensor(w2v[r'<\s>'])
            elif id2word[i] in w2v:
                embedding[i, :] = torch.Tensor(w2v[id2word[i]])
            else:
                oovs.append(i)
    return embedding, oovs

def count_data_stat(path):
    """ count statistics of the data"""
    max_article_split = []
    max_sentence_split = []
    median_article_split = []
    median_sentence_split = []
    for split in ['train', 'val', 'test']:
        art_sents = []
        data_path = join(path, split)
        for i in range(count_data(data_path)):
            with open(join(data_path, '{}.json'.format(i))) as f:
                js = json.loads(f.read())
                art_sents.append(js['article'])
        article_size = [len(story) for story in art_sents] #number of sentences in an article
        sentence_size = [len(row) for row in chain.from_iterable([story for story in art_sents])] #max number of words in a sentence
        max_article_size = max(article_size)
        max_sentence_size = max(sentence_size)
        median_article_size = median(article_size)
        median_sentence_size = median(sentence_size)
        max_article_split.append(max_article_size)
        max_sentence_split.append(max_sentence_size)
        median_article_split.append(median_article_size)
        median_sentence_split.append(median_sentence_size)
        print('######## Statistics for', split,'split: ######')
        print('Number of data:', count_data(data_path))
        print('Max number of sentences in an article:', max_article_size)
        print('Median number of sentences in an article:', median_article_size)
        print('Max number of words in a sentence:', max_sentence_size)
        print('Median number of words in a sentence:', median_sentence_size)

    return max(max_article_split), max(max_sentence_split), max(median_article_split), max(median_sentence_split)

#MAX_ARTICLE_SIZE, MAX_SENTENCE_SIZE, MEAN_ARTICLE_SIZE, MEAN_SENTENCE_SIZE = count_data_stat(DATA_DIR)
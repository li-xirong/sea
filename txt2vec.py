import os
import numpy as np
import pickle
from bigfile import BigFile
from common import logger
from textlib import TextTool
# from infersent import InferSent
import torch


def get_lang(data_path):
    return 'en'


class Txt2Vec(object):
    '''
    norm: 0 no norm, 1 l_1 norm, 2 l_2 norm
    '''
    def __init__(self, data_path, norm=0, clean=True):
        logger.info(self.__class__.__name__ + ' initializing ...')
        self.data_path = data_path
        self.norm = norm
        self.lang = get_lang(data_path)
        self.clean = clean
        assert (norm in [0, 1, 2]), 'invalid norm %s' % norm

    def _preprocess(self, query):
        words = TextTool.tokenize(query, clean=self.clean, language=self.lang)
        return words

    def _do_norm(self, vec):
        assert (1 == self.norm or 2 == self.norm)
        norm = np.linalg.norm(vec, self.norm)
        return vec / (norm + 1e-10)  # avoid divide by ZERO

    def _encoding(self, words):
        raise Exception("encoding not implemented yet!")

    def encoding(self, query):
        words = self._preprocess(query)
        vec = self._encoding(words)
        if self.norm > 0:
            return self._do_norm(vec)
        return vec


class BowVec(Txt2Vec):
    def __init__(self, data_path, norm=0, clean=True):
        super(BowVec, self).__init__(data_path, norm, clean)
        self.vocab = pickle.load(open(data_path, 'rb'))
        self.ndims = len(self.vocab)
        logger.info('vob size: %d, vec dim: %d' %
                    (len(self.vocab), self.ndims))

    def _encoding(self, words):
        vec = np.zeros(self.ndims, )

        for word in words:
            idx = self.vocab.find(word)
            if idx >= 0:
                vec[idx] += 1
        return vec


class SoftBowVec(Txt2Vec):
    def __init__(self, data_path, thresh=5, norm=0, clean=True):
        super(SoftBowVec, self).__init__(data_path, norm, clean)
        self.vocab = pickle.load(open(data_path, 'rb'))
        self.ndims = len(self.vocab)
        self.thresh = thresh
        logger.info('vob size: %d, vec dim: %d' %
                    (len(self.vocab), self.ndims))

    def _encoding(self, words):
        vec = np.zeros(self.ndims, )

        for word in words:
            idx = self.vocab.find(word)
            if idx >= 0:
                vec[idx] += 1
                for w, s in self.vocab.word2sim[word][1:self.thresh + 1]:
                    vec[self.vocab(w)] += s
        return vec


class SoftBowVecNSW(SoftBowVec):
    def __init__(self, data_path, thresh=5, norm=0, clean=True):
        super(SoftBowVecNSW, self).__init__(data_path, thresh, norm, clean)
        if '_nsw' not in data_path:
            logger.error(
                'WARNING: loaded a vocabulary that contains stopwords')

    def _preprocess(self, query):
        words = TextTool.tokenize(query,
                                  clean=self.clean,
                                  language=self.lang,
                                  remove_stopword=True)
        return words


class W2Vec(Txt2Vec):
    def __init__(self, data_path, norm=0, clean=True):
        super(W2Vec, self).__init__(data_path, norm, clean)
        self.w2v = BigFile(data_path)
        vocab_size, self.ndims = self.w2v.shape()
        logger.info('vob size: %d, vec dim: %d' % (vocab_size, self.ndims))

    def _encoding(self, words):
        renamed, vectors = self.w2v.read(words)

        if len(vectors) > 0:
            vec = np.array(vectors).mean(axis=0)
        else:
            vec = np.zeros(self.ndims, )
        return vec

    def raw_encoding(self, query):
        words = self._preprocess(query)
        renamed, vectors = self.w2v.read(words)

        if len(vectors) > 0:
            vec = np.array(vectors)
        else:
            vec = np.zeros((len(words), self.ndims))
        return vec


class InferVec(Txt2Vec):
    def __init__(self, model_path, norm=0, clean=True):
        super(InferVec, self).__init__(model_path, norm, clean)
        self.ndims = 4096

        params_model = {
            'bsize': 64,
            'word_emb_dim': 300,
            'enc_lstm_dim': 2048,
            'pool_type': 'max',
            'dpout_model': 0.0,
            'version': 2
        }
        model = InferSent(params_model)
        model.load_state_dict(torch.load(model_path))

        use_cuda = True
        self.model = model.cuda() if use_cuda else model

        #W2V_PATH = 'glove/glove.840B.300d.txt' if model_version == 1 else 'fasttext/crawl-300d-2M.vec'
        names = os.path.basename(os.path.splitext(model_path)[0]).split('_')
        data_path = os.path.join(os.path.dirname(model_path), *names[1:])
        self.model.set_w2v_path(data_path)

        self.model.build_vocab_k_words(K=100000)

        #logger.info('vob size: %d, vec dim: %d' % (vocab_size, self.ndims))

    def _encoding(self, words):
        vec = self.model.encode(words, tokenize=True)
        vec = vec.reshape(self.ndims, )

        return vec

    def encoding(self, query):
        sentences = [query]
        vec = self._encoding(sentences)

        return vec


class IndexVec(Txt2Vec):
    def __init__(self, data_path, clean=True):
        super(IndexVec, self).__init__(data_path, 0, clean)
        self.vocab = pickle.load(open(data_path, 'rb'))
        logger.info('vob size: %s' % (len(self.vocab)))

    def _preprocess(self, query):
        words = TextTool.tokenize(query,
                                  clean=self.clean,
                                  language=self.lang,
                                  remove_stopword=False)
        words = ['<start>'] + words + ['<end>']
        return words

    def _encoding(self, words):
        return np.array([self.vocab(word) for word in words])


class BowVecNSW(BowVec):
    def __init__(self, data_path, norm=0, clean=True):
        super(BowVecNSW, self).__init__(data_path, norm, clean)
        if '_nsw' not in data_path:
            logger.error(
                'WARNING: loaded a vocabulary that contains stopwords')

    def _preprocess(self, query):
        words = TextTool.tokenize(query,
                                  clean=self.clean,
                                  language=self.lang,
                                  remove_stopword=True)
        return words


class W2VecNSW(W2Vec):
    def _preprocess(self, query):
        words = TextTool.tokenize(query,
                                  clean=self.clean,
                                  language=self.lang,
                                  remove_stopword=True)
        return words


class PrecomputedSentFeat(object):
    def __init__(self, collection, sent_feat_name, rootpath):
        logger.info(self.__class__.__name__ + ' initializing ...')
        sent_feat_file_path = os.path.join(rootpath, collection, 'TextData',
                                           'PrecomputedSentFeat',
                                           sent_feat_name)
        self.s2v = BigFile(sent_feat_file_path)
        vocab_size, self.ndims = self.s2v.shape()
        logger.info('vob size: %d, vec dim: %d' % (vocab_size, self.ndims))

    def encoding(self, cap_ids):
        """Convert ids of captions to corresponding precomputed features.
        
        Note that the paramete is 'cap_ids' instead of 'query'.
        """
        return [self.s2v.read_one(cap_id) for cap_id in cap_ids]
        # return [ [ 0 for i in range(768)] for cap_id in cap_ids]
        # _, vectors = self.s2v.read(cap_ids)
        # return vectors


NAME_TO_T2V = {
    'bow': BowVec,
    'bow_nsw': BowVecNSW,
    'soft_bow': SoftBowVec,
    'soft_bow_nsw': SoftBowVecNSW,
    'w2v': W2Vec,
    'w2v_nsw': W2VecNSW,
    'idxvec': IndexVec,
    'infersent': InferVec,
    'precomputed_sent_feature': PrecomputedSentFeat,
}


def get_txt2vec(name):
    assert name in NAME_TO_T2V
    return NAME_TO_T2V[name]


if __name__ == '__main__':
    # t2v = BowVec('VisualSearch/tgif-msrvtt10k/TextData/vocab/bow_5.pkl')
    # t2v = BowVecNSW('VisualSearch/tgif-msrvtt10k/TextData/vocab/bow_nsw_5.pkl')
    # t2v = BowVecNSW('VisualSearch/tgif-msrvtt10k/TextData/vocab/bow_5.pkl')
    # t2v = W2Vec('VisualSearch/word2vec/flickr/vec500flickr30m')
    # t2v = W2VecNSW('VisualSearch/word2vec/flickr/vec500flickr30m')
    # t2v = SoftBowVecNSW('VisualSearch/tgif-msrvtt10k/TextData/vocab/soft_bow_nsw_5.pkl')
    # t2v = InferVec('VisualSearch/encoder/infersent_fasttext_crawl-300d-2M.vec.pkl')
    # vec = t2v.encoding('a dog runs on grass')
    # print vec.shape
    # print [(t2v.vocab[i],j) for i, j in enumerate(vec) if j != 0]

    s2v = PrecomputedSentFeat('msvdtrain',
                              'bert_feature_Layer_-2_uncased_L-12_H-768_A-12',
                              'data')
    vecs = s2v.encoding(
        ['ZbzDGXEwtGc_6_15#0', 'ZbzDGXEwtGc_6_15#1', 'ZbzDGXEwtGc_6_15#2'])
    print(vecs)
    print(len(vecs))

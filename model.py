import copy
import time
from collections import OrderedDict

import torch
import numpy as np
import torch.nn as nn
import torch.nn.init
import torch.backends.cudnn as cudnn
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


import util
from aggregator import *
from bigfile import BigFile
from common import logger
from generic_utils import Progbar
from evaluation import compute_sim
from loss import cosine_sim, MarginRankingLoss, MarginRankingLoss_adv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_we(vocab, w2v_dir):
    w2v = BigFile(w2v_dir)
    ndims = w2v.ndims
    nr_words = len(vocab)
    words = [vocab[i] for i in range(nr_words)]
    we = np.random.uniform(low=-1.0, high=1.0, size=(nr_words, ndims))

    renamed, vecs = w2v.read(words)
    for i, word in enumerate(renamed):
        idx = vocab.find(word)
        we[idx] = vecs[i]

    return torch.Tensor(we)


def l2norm(X):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=1, keepdim=True).sqrt()
    X = torch.div(X, norm+1e-10)
    return X


def _initialize_weights(m):
    """Initialize module weights
    """
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif type(m) == nn.BatchNorm1d:
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)


class IdentityNet(nn.Module):
    def __init__(self, opt):
        super(IdentityNet, self).__init__()

    def forward(self, input_x):
        """Extract image feature vectors."""
        return input_x.to(device)


class TransformNet(nn.Module):
    def __init__(self, fc_layers, opt):
        super(TransformNet, self).__init__()

        self.fc1 = nn.Linear(fc_layers[0], fc_layers[1])
        if opt.batch_norm:
            self.bn1 = nn.BatchNorm1d(fc_layers[1])
        else:
            self.bn1 = None

        if opt.activation == 'tanh':
            self.activation = nn.Tanh()
        elif opt.activation == 'relu':
            self.activation = nn.ReLU()
        else:
            self.activation = None

        if opt.dropout > 1e-3:
            self.dropout = nn.Dropout(p=opt.dropout)
        else:
            self.dropout = None

        self.init_weights()

    def init_weights(self):
        """Xavier initialization for the fully connected layer
        """
        self.apply(_initialize_weights)


    def forward(self, input_x):
        features = self.fc1(input_x.to(device))

        if self.bn1 is not None:
            features = self.bn1(features)

        if self.activation is not None:
            features = self.activation(features)

        if self.dropout is not None:
            features = self.dropout(features)

        return features

    def load_state_dict(self, state_dict):
        """Copies parameters. overwritting the default one to
        accept state_dict from Full model
        """
        own_state = self.state_dict()
        new_state = OrderedDict()
        for name, param in state_dict.items():
            if name in own_state:
                new_state[name] = param

        super(TransformNet, self).load_state_dict(new_state)


class VisTransformNet (TransformNet):
    def __init__(self, opt):
        super(VisTransformNet, self).__init__(opt.vis_fc_layers, opt)


class VisNet(nn.Module):
    def _init_encoder(self, opt):
        self.encoder = IdentityNet(opt)

    def _init_transformer(self, opt):
        self.transformer = VisTransformNet(opt)

    def __init__(self, opt):
        super(VisNet, self).__init__()
        self._init_encoder(opt)
        self._init_transformer(opt)

    def forward(self, vis_input):
        features = self.encoder(vis_input)
        features = self.transformer(features)
        return features

    def load_state_dict(self, state_dict):
        #own_state = self.state_dict()
        #for name, param in own_state.items():
        #    own_state[name] = state_dict[name.split('.',1)[1]]

        #super(VisNet, self).load_state_dict(own_state)
        super(VisNet, self).load_state_dict(state_dict)


class AvgPoolVisNet(VisNet):
    def _init_encoder(self, opt):
        self.encoder = AvgPooling()


class NetVLADVisNet(VisNet):
    def _init_encoder(self, opt):
        self.encoder = NetVLAD_AvgPooling(opt)


class TxtTransformNet (TransformNet):
    def __init__(self, opt):
        super(TxtTransformNet, self).__init__(opt.txt_fc_layers, opt)


class TxtEncoder(nn.Module):
    def __init__(self, opt):
        super(TxtEncoder, self).__init__()

    def forward(self, txt_input, cap_ids):
        return txt_input


class GruTxtEncoder(TxtEncoder):
    def _init_rnn(self, opt):
        self.rnn = nn.GRU(opt.we_dim, opt.rnn_size, opt.rnn_layer, batch_first=True, dropout=opt.rnn_dropout, bidirectional=False)

    def __init__(self, opt):
        super(GruTxtEncoder, self).__init__(opt)
        self.pooling = opt.pooling
        self.t2v_idx = opt.t2v_idx
        self.we = nn.Embedding(len(self.t2v_idx.vocab), opt.we_dim)
        if opt.we_dim == 500:
            self.we.weight = nn.Parameter(opt.we) # initialize with a pre-trained 500-dim w2v

        self._init_rnn(opt)
        self.rnn_size = opt.rnn_size

    def forward(self, txt_input, cap_ids):
        """Handles variable size captions
        """
        batch_size = len(txt_input)

        # caption encoding
        idx_vecs = [self.t2v_idx.encoding(caption) for caption in txt_input]
        lengths = [len(vec) for vec in idx_vecs]

        x = torch.zeros(batch_size, max(lengths)).long().to(device)
        for i, vec in enumerate(idx_vecs):
            end = lengths[i]
            x[i, :end] = torch.Tensor(vec)

        # caption embedding
        x = self.we(x)
        packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)

        # Forward propagate RNN
        out, _ = self.rnn(packed)
        # Reshape *final* output to (batch_size, hidden_size)
        padded = pad_packed_sequence(out, batch_first=True)

        if self.pooling == 'mean':
            out = torch.zeros(batch_size, self.rnn_size).to(device)
            for i, ln in enumerate(lengths):
                out[i] = torch.mean(padded[0][i][:ln], dim=0)
        elif self.pooling == 'last':
            I = torch.LongTensor(lengths).view(-1, 1, 1)
            I = I.expand(batch_size, 1, self.rnn_size) - 1
            I = I.cuda()
            out = torch.gather(padded[0], 1, I).squeeze(1)
        elif self.pooling == 'mean_last':
            out1 = torch.zeros(batch_size, self.rnn_size).to(device)
            for i, ln in enumerate(lengths):
                out1[i] = torch.mean(padded[0][i][:ln], dim=0)

            I = torch.LongTensor(lengths).view(-1, 1, 1)
            I = I.expand(batch_size, 1, self.rnn_size) - 1
            I = I.cuda()
            out2 = torch.gather(padded[0], 1, I).squeeze(1)
            out = torch.cat((out1, out2), dim=1)
        return out


class LstmTxtEncoder(GruTxtEncoder):
    def _init_rnn(self, opt):
        self.rnn = nn.LSTM(opt.we_dim, opt.rnn_size, opt.rnn_layer, batch_first=True, dropout=opt.rnn_dropout, bidirectional=False)


class BiGruTxtEncoder(GruTxtEncoder):
    def _init_rnn(self, opt):
        self.rnn = nn.GRU(opt.we_dim, opt.rnn_size, opt.rnn_layer, batch_first=True, dropout=opt.rnn_dropout, bidirectional=True)

    def __init__(self, opt):
        super(BiGruTxtEncoder, self).__init__(opt)
        self.rnn_size = opt.rnn_size*2


class BoWTxtEncoder (TxtEncoder):
    def __init__(self, opt):
        super(BoWTxtEncoder, self).__init__(opt)
        self.t2v_bow = opt.t2v_bow
        if hasattr(opt, 'bow_encoder_l2norm'):
            self.bow_encoder_l2norm = opt.bow_encoder_l2norm
        else:
            self.bow_encoder_l2norm = False

    def forward(self, txt_input, cap_ids):
        bow_out = torch.Tensor([self.t2v_bow.encoding(caption) for caption in txt_input]).to(device)
        if self.bow_encoder_l2norm:
            bow_out = l2norm(bow_out)
        return bow_out

class SoftBoWTxtEncoder (BoWTxtEncoder):
    def __init__(self, opt):
        super(SoftBoWTxtEncoder, self).__init__(opt)
        self.t2v_bow = opt.t2v_bow


class W2VTxtEncoder (TxtEncoder):
    def __init__(self, opt):
        super(W2VTxtEncoder, self).__init__(opt)
        self.is_online, self.is_precomputed = False, False
        for encoding in opt.text_encoding.split('@'):
            if 'precomputed_w2v' in encoding:
                self.id2v = opt.precomputed_feat_w2v
                self.is_precomputed = True
                logger.info('Offline Word2vec initializing')
            elif 'w2v' in encoding:
                self.t2v_w2v = opt.t2v_w2v             
                self.is_online = True
                logger.info('Online Word2vec model initializing')
            
        assert(self.is_online or self.is_precomputed)     

    def forward(self, txt_input, cap_ids):
        if self.is_precomputed:
            cap_ids = list(cap_ids)
            w2v_out = torch.Tensor( self.id2v.encoding(cap_ids)).to(device)
        else:
            w2v_out = torch.Tensor([self.t2v_w2v.encoding(caption) for caption in txt_input]).to(device)
        return w2v_out


class W2V_NetVLADTxtEncoder (TxtEncoder):
    def __init__(self, opt):
        super(W2V_NetVLADTxtEncoder, self).__init__(opt)
        self.t2v_w2v = opt.t2v_w2v
        self.netvlad = NetVLAD(opt)

    def forward(self, txt_input, cap_ids):
        w2v_out = [torch.Tensor(self.t2v_w2v.raw_encoding(caption)) for caption in txt_input]
        w2v_out = self.netvlad(w2v_out)

        return w2v_out

class InfersentTxtEncoder (TxtEncoder):
    def __init__(self, opt):
        super(InfersentTxtEncoder, self).__init__(opt)
        self.t2v_infer = opt.t2v_infer

    def forward(self, txt_input, cap_ids):
        infersent_out = torch.Tensor([self.t2v_infer.encoding(caption) for caption in txt_input]).to(device)
        return infersent_out

class BertTxtEncoder (TxtEncoder):
    def __init__(self, opt):
        super(BertTxtEncoder, self).__init__(opt)
        self.is_online, self.is_precomputed = False, False

        for encoding in opt.text_encoding.split('@'):
            if 'online_bert' in encoding:
                from bert_serving.client import BertClient
                self.bc = BertClient(ip='10.77.50.197', port=1234, port_out=4321, check_version=False)
                self.is_online = True
                logger.info('Online BertClient initializing')

            if 'precomputed_bert' in encoding:
                self.id2v = opt.precomputed_feat_bert
                self.is_precomputed = True
                logger.info('Offline bert initializing')
        assert(self.is_online or self.is_precomputed)


    def forward(self, txt_input, cap_ids):
        if self.is_precomputed:
            cap_ids = list(cap_ids)
            bert_out = torch.Tensor( self.id2v.encoding(cap_ids)).to(device)
        else:
            txt_input = list(txt_input)
            bert_out = torch.Tensor( self.bc.encode(txt_input) ).to(device)
        return bert_out


class MultiScaleTxtEncoder_w2v_bow (TxtEncoder):
    def __init__(self, opt):
        super(MultiScaleTxtEncoder_w2v_bow, self).__init__(opt)
        self.w2v_encoder = W2VTxtEncoder(opt)
        self.bow_encoder = BoWTxtEncoder(opt)

    def forward(self, txt_input, cap_ids):
        w2v_out = self.w2v_encoder(txt_input, cap_ids)
        bow_out = self.bow_encoder(txt_input, cap_ids)

        out = torch.cat((w2v_out, bow_out), dim=1)
        return out

class MultiScaleTxtEncoder_bert_w2v_bow (MultiScaleTxtEncoder_w2v_bow):
    def __init__(self, opt):
        super(MultiScaleTxtEncoder_bert_w2v_bow, self).__init__(opt)
        self.bert_encoder = BertTxtEncoder(opt)

    def forward(self, txt_input, cap_ids):
        w2v_out = self.w2v_encoder(txt_input, cap_ids)
        bow_out = self.bow_encoder(txt_input, cap_ids)
        bert_out = self.bert_encoder(txt_input, cap_ids)
        out = torch.cat((bert_out, w2v_out, bow_out), dim=1)
        return out




class MultiScaleTxtEncoder_netvlad_bow (TxtEncoder):
    def __init__(self, opt):
        super(MultiScaleTxtEncoder_netvlad_bow, self).__init__(opt)
        self.netvlad_encoder = W2V_NetVLADTxtEncoder(opt)
        self.bow_encoder = BoWTxtEncoder(opt)
        self.txt_cat_norm = opt.txt_cat_norm
        print('txt_cat_norm: %s' % self.txt_cat_norm)

    def forward(self, txt_input, cap_ids):
        netvlad_out = self.netvlad_encoder(txt_input, cap_ids)
        bow_out = self.bow_encoder(txt_input, cap_ids)

        if self.txt_cat_norm:
            bow_out = l2norm(bow_out)

        out = torch.cat((netvlad_out, bow_out), dim=1)
        return out

class MultiScaleTxtEncoder_bigru_netvlad_bow (TxtEncoder):
    def __init__(self, opt):
        super(MultiScaleTxtEncoder_bigru_netvlad_bow, self).__init__(opt)
        self.rnn_encoder = BiGruTxtEncoder(opt)
        self.netvlad_encoder = W2V_NetVLADTxtEncoder(opt)
        self.bow_encoder = BoWTxtEncoder(opt)
        self.txt_cat_norm = opt.txt_cat_norm
        print('txt_cat_norm: %s' % self.txt_cat_norm)

    def forward(self, txt_input, cap_ids):
        rnn_out = self.rnn_encoder(txt_input, cap_ids)
        netvlad_out = self.netvlad_encoder(txt_input, cap_ids)
        bow_out = self.bow_encoder(txt_input, cap_ids)

        if self.txt_cat_norm:
            rnn_out = l2norm(rnn_out)
            bow_out = l2norm(bow_out)

        out = torch.cat((rnn_out, netvlad_out, bow_out), dim=1)
        return out

class MultiScaleTxtEncoder_w2v_softbow(MultiScaleTxtEncoder_w2v_bow):
    def __init__(self, opt):
        super(MultiScaleTxtEncoder_w2v_softbow, self).__init__(opt)
        self.bow = SoftBoWTxtEncoder(opt)


class MultiScaleTxtEncoder_gru_w2v_bow (TxtEncoder):
    def __init__(self, opt):
        super(MultiScaleTxtEncoder_gru_w2v_bow, self).__init__(opt)
        self.rnn_encoder = GruTxtEncoder(opt)
        self.w2v_encoder = W2VTxtEncoder(opt)
        self.bow_encoder = BoWTxtEncoder(opt)

    def forward(self, txt_input, cap_ids):
        """Handles variable size captions
        """
        # Embed word ids to vectors
        rnn_out = self.rnn_encoder(txt_input, cap_ids)
        w2v_out = self.w2v_encoder(txt_input, cap_ids)
        bow_out = self.bow_encoder(txt_input, cap_ids)

        out = torch.cat((rnn_out, w2v_out, bow_out), dim=1)
        return out


class MultiScaleTxtEncoder_bert_gru_w2v_bow (MultiScaleTxtEncoder_gru_w2v_bow):
    def __init__(self, opt):
        super(MultiScaleTxtEncoder_bert_gru_w2v_bow, self).__init__(opt)
        self.bert_encoder = BertTxtEncoder(opt)

    def forward(self, txt_input, cap_ids):
        """Handles variable size captions
        """
        # Embed word ids to vectors
        rnn_out = self.rnn_encoder(txt_input, cap_ids)
        w2v_out = self.w2v_encoder(txt_input, cap_ids)
        bow_out = self.bow_encoder(txt_input, cap_ids)
        bert_out = self.bert_encoder(txt_input, cap_ids)
        out = torch.cat((bert_out, rnn_out, w2v_out, bow_out), dim=1)
        return out


class MultiScaleTxtEncoder_lstm_w2v_bow (MultiScaleTxtEncoder_gru_w2v_bow):
    def __init__(self, opt):
        super(MultiScaleTxtEncoder_lstm_w2v_bow, self).__init__(opt)
        self.rnn_encoder = LstmTxtEncoder(opt)

class MultiScaleTxtEncoder_bigru_w2v_bow(MultiScaleTxtEncoder_gru_w2v_bow):
    def __init__(self, opt):
        super(MultiScaleTxtEncoder_bigru_w2v_bow, self).__init__(opt)
        self.rnn_encoder = BiGruTxtEncoder(opt)



class MultiScaleTxtEncoder_bert_bigru_w2v_bow(MultiScaleTxtEncoder_bigru_w2v_bow):
    def __init__(self, opt):
        super(MultiScaleTxtEncoder_bert_bigru_w2v_bow, self).__init__(opt)
        self.bert_encoder = BertTxtEncoder(opt)

    def forward (self, txt_input, cap_ids):
        bert_out = self.bert_encoder(txt_input, cap_ids)
        rnn_out = self.rnn_encoder(txt_input, cap_ids)
        w2v_out = self.w2v_encoder(txt_input, cap_ids)
        bow_out = self.bow_encoder(txt_input, cap_ids)

        out = torch.cat((bert_out, rnn_out, w2v_out, bow_out), dim=1)
        return out


class MultiScaleTxtEncoder_gruFC_w2vFC_bowFC (TxtEncoder):
    def __init__(self, opt):
        super(MultiScaleTxtEncoder_gruFC_w2vFC_bowFC, self).__init__(opt)
        self.rnn_encoder = GruTxtEncoder(opt)
        rnn_opt = copy.deepcopy(opt)
        rnn_opt.txt_fc_layers[0] = opt.rnn_size
        self.rnn_fc = TxtTransformNet(rnn_opt)

        self.w2v_encoder = W2VTxtEncoder(opt)
        w2v_opt = copy.deepcopy(opt)
        w2v_opt.txt_fc_layers[0] = opt.w2v_out_size
        self.w2v_fc = TxtTransformNet(w2v_opt)

        self.bow_encoder = BoWTxtEncoder(opt)
        bow_opt = copy.deepcopy(opt)
        bow_opt.txt_fc_layers[0] = opt.t2v_bow.ndims
        self.bow_fc = TxtTransformNet(bow_opt)

        self.out_size = opt.txt_fc_layers[-1]*3

    def forward(self, txt_input, cap_ids):
        # Embed word ids to vectors
        rnn_out = self.rnn_fc(self.rnn_encoder(txt_input, cap_ids))
        w2v_out = self.w2v_fc(self.w2v_encoder(txt_input, cap_ids))
        bow_out = self.bow_fc(self.bow_encoder(txt_input, cap_ids))

        out = torch.cat((rnn_out, w2v_out, bow_out), dim=1)
        return out

class MultiScaleTxtEncoder_gru_w2v_bowFC (TxtEncoder):
    def __init__(self, opt):
        super(MultiScaleTxtEncoder_gru_w2v_bowFC, self).__init__(opt)
        self.rnn_encoder = GruTxtEncoder(opt)
        self.w2v_encoder = W2VTxtEncoder(opt)

        self.bow_encoder = BoWTxtEncoder(opt)
        bow_opt = copy.deepcopy(opt)
        bow_opt.txt_fc_layers[0] = opt.t2v_bow.ndims
        self.bow_fc = TxtTransformNet(bow_opt)

        self.out_size = opt.rnn_size + opt.w2v_out_size + opt.txt_fc_layers[-1]

    def forward(self, txt_input, cap_ids):
        rnn_out =  self.rnn_encoder(txt_input, cap_ids)
        w2v_out =  self.w2v_encoder(txt_input, cap_ids)
        bow_out = self.bow_fc(self.bow_encoder(txt_input, cap_ids))

        out = torch.cat((rnn_out, w2v_out, bow_out), dim=1)
        return out

class MultiScaleTxtEncoder_gru_w2v_softbow (MultiScaleTxtEncoder_gru_w2v_bow):
    def __init__(self, opt):
        super(MultiScaleTxtEncoder_gru_w2v_softbow, self).__init__(opt)
        self.bow_encoder = SoftBoWTxtEncoder(opt)

class MultiScaleTxtEncoder_infersent_w2v_bow (TxtEncoder):
    def __init__(self, opt):
        super(MultiScaleTxtEncoder_infersent_w2v_bow, self).__init__(opt)
        self.inf_encoder = InfersentTxtEncoder(opt)
        self.w2v_encoder = W2VTxtEncoder(opt)
        self.bow_encoder = BoWTxtEncoder(opt)

    def forward(self, txt_input, cap_ids):
        inf_out = self.inf_encoder(txt_input, cap_ids)
        w2v_out = self.w2v_encoder(txt_input, cap_ids)
        bow_out = self.bow_encoder(txt_input, cap_ids)

        out = torch.cat((inf_out, w2v_out, bow_out), dim=1)
        return out

class MultiScaleTxtEncoder_infersent_w2v_softbow(MultiScaleTxtEncoder_infersent_w2v_bow):
    def __init__(self, opt):
        super(MultiScaleTxtEncoder_infersent_w2v_softbow, self).__init__(opt)
        self.bow_encoder = SoftBoWTxtEncoder(opt)


class TxtNet (nn.Module):
    def _init_encoder(self, opt):
        self.encoder = TxtEncoder(opt)

    def _init_transformer(self, opt):
        self.transformer = TxtTransformNet(opt)

    def __init__(self, opt):
        super(TxtNet, self).__init__()
        self._init_encoder(opt)
        self._init_transformer(opt)

    def forward(self, txt_input, cap_ids):
        features = self.encoder(txt_input, cap_ids)
        features = self.transformer(features)
        return features


class BoWTxtNet(TxtNet):
    def _init_encoder(self, opt):
        opt.txt_fc_layers[0] = opt.t2v_bow.ndims
        self.encoder = BoWTxtEncoder(opt)


class SoftBoWTxtNet(TxtNet):
    def _init_encoder(self, opt):
        opt.txt_fc_layers[0] = opt.t2v_bow.ndims
        self.encoder = SoftBoWTxtEncoder(opt)


class W2VTxtNet(TxtNet):
    def _init_encoder(self, opt):
        opt.txt_fc_layers[0] = opt.w2v_out_size
        self.encoder = W2VTxtEncoder(opt)

class W2V_NetVLADTxtNet(TxtNet):
    def _init_encoder(self, opt):
        opt.txt_fc_layers[0] = opt.netvlad_num_clusters * opt.feature_dim
        self.encoder = W2V_NetVLADTxtEncoder(opt)

class InfersentTxtNet(TxtNet):
    def _init_encoder(self, opt):
        opt.txt_fc_layers[0] = opt.t2v_infer.ndims
        self.encoder = InfersentTxtEncoder(opt)
 
class GruTxtNet(TxtNet):
    def _init_encoder(self, opt):
        opt.txt_fc_layers[0] = opt.rnn_size
        self.encoder = GruTxtEncoder(opt)

class LstmTxtNet(TxtNet):
    def _init_encoder(self, opt):
        opt.txt_fc_layers[0] = opt.rnn_size
        self.encoder = LstmTxtEncoder(opt)


class GruTxtNet_mean_last(GruTxtNet):
    def _init_encoder(self, opt):
        opt.txt_fc_layers[0] = opt.rnn_size*2
        assert opt.pooling == 'mean_last', 'Gru pooling type(%s) not mathed.'%opt.pooling
        self.encoder = GruTxtEncoder(opt)

class BiGruTxtNet(GruTxtNet):
    def _init_encoder(self, opt):
        opt.txt_fc_layers[0] = opt.rnn_size*2
        self.encoder = BiGruTxtEncoder(opt)

class BertTxtNet(TxtNet):
    def _init_encoder(self, opt):
        opt.txt_fc_layers[0] = opt.bert_out_size
        self.encoder = BertTxtEncoder(opt)


class MultiScaleTxtNet_w2v_bow (TxtNet):
    def _init_encoder(self, opt):
        opt.txt_fc_layers[0] = opt.t2v_bow.ndims + opt.w2v_out_size
        self.encoder = MultiScaleTxtEncoder_w2v_bow(opt)

class MultiScaleTxtNet_netvlad_bow (TxtNet):
    def _init_encoder(self, opt):
        opt.txt_fc_layers[0] = opt.netvlad_num_clusters * opt.feature_dim + opt.t2v_bow.ndims
        self.encoder = MultiScaleTxtEncoder_netvlad_bow(opt)

class MultiScaleTxtNet_bigru_netvlad_bow (TxtNet):
    def _init_encoder(self, opt):
        opt.txt_fc_layers[0] = opt.rnn_size*2 + opt.t2v_bow.ndims + opt.netvlad_num_clusters * opt.feature_dim
        self.encoder = MultiScaleTxtEncoder_bigru_netvlad_bow(opt)

class MultiScaleTxtNet_gru_w2v_bow (TxtNet):
    def _init_encoder(self, opt):
        opt.txt_fc_layers[0] = (opt.rnn_size + opt.t2v_bow.ndims + opt.w2v_out_size)
        self.encoder = MultiScaleTxtEncoder_gru_w2v_bow(opt)

class MultiScaleTxtNet_lstm_w2v_bow (TxtNet):
    def _init_encoder(self, opt):
        opt.txt_fc_layers[0] = (opt.rnn_size + opt.t2v_bow.ndims + opt.w2v_out_size)
        self.encoder = MultiScaleTxtEncoder_lstm_w2v_bow(opt)

class MultiScaleTxtNet_gru_w2v_softbow(TxtNet):
    def _init_encoder(self, opt):
        opt.txt_fc_layers[0] = (opt.rnn_size + opt.t2v_bow.ndims + opt.w2v_out_size)
        self.encoder = MultiScaleTxtEncoder_gru_w2v_softbow(opt)

class MultiScaleTxtNet_infersent_w2v_bow(TxtNet):
    def _init_encoder(self, opt):
        opt.txt_fc_layers[0] = opt.t2v_infer.ndims + opt.t2v_bow.ndims + opt.w2v_out_size
        self.encoder = MultiScaleTxtEncoder_infersent_w2v_bow(opt)

class MultiScaleTxtNet_infersent_w2v_softbow(TxtNet):
    def _init_encoder(self, opt):
        opt.txt_fc_layers[0] = opt.t2v_infer.ndims + opt.t2v_bow.ndims + opt.w2v_out_size
        self.encoder = MultiScaleTxtEncoder_infersent_w2v_softbow(opt)

class MultiScaleTxtNet_bigru_w2v_bow(TxtNet):
    def _init_encoder(self, opt):
        opt.txt_fc_layers[0] = opt.rnn_size*2 + opt.t2v_bow.ndims + opt.w2v_out_size
        self.encoder = MultiScaleTxtEncoder_bigru_w2v_bow(opt)

class MultiScaleTxtNet_bert_bigru_w2v_bow(TxtNet):
    def _init_encoder(self, opt):
        opt.txt_fc_layers[0] = opt.rnn_size*2 + opt.t2v_bow.ndims + opt.w2v_out_size + opt.bert_out_size
        self.encoder = MultiScaleTxtEncoder_bert_bigru_w2v_bow(opt)

class MultiScaleTxtNet_bert_gru_w2v_bow(TxtNet):
    def _init_encoder(self, opt):
        opt.txt_fc_layers[0] = opt.rnn_size + opt.t2v_bow.ndims + opt.w2v_out_size + opt.bert_out_size
        self.encoder = MultiScaleTxtEncoder_bert_gru_w2v_bow(opt)

class MultiScaleTxtNet_bert_w2v_bow(TxtNet):
    def _init_encoder(self, opt):
        opt.txt_fc_layers[0] = opt.t2v_bow.ndims + opt.w2v_out_size + opt.bert_out_size
        self.encoder = MultiScaleTxtEncoder_bert_w2v_bow(opt)






class MultiScaleTxtNet_gruFC_w2vFC_bowFC(TxtNet):
    def _init_encoder(self, opt):
        self.encoder = MultiScaleTxtEncoder_gruFC_w2vFC_bowFC(opt)
        opt.txt_fc_layers[0] = self.encoder.out_size

class MultiScaleTxtNet_gru_w2v_bowFC(TxtNet):
    def _init_encoder(self, opt):
        self.encoder = MultiScaleTxtEncoder_gru_w2v_bowFC(opt)
        opt.txt_fc_layers[0] = self.encoder.out_size


class MultiSpaceTxtNet (nn.Module):
    def __init__(self, opt):
        super(MultiSpaceTxtNet, self).__init__()
        self.txt_net_m = MultiScaleTxtNet_gru_w2v_bow(opt)

        self.txt_net_b = BoWTxtNet(opt)

        self.txt_net_w = W2VTxtNet(opt)

        self.txt_net_g = GruTxtNet(opt)

    def forward(self, txt_input, cap_ids):
        feature_m = self.txt_net_m(txt_input, cap_ids)
        feature_b = self.txt_net_b(txt_input, cap_ids)
        feature_w = self.txt_net_w(txt_input, cap_ids)
        feature_g = self.txt_net_g(txt_input, cap_ids)

        return feature_m, feature_b, feature_w, feature_g


class MultiSpaceTxtNet_bow_w2v_gru (nn.Module):
    def __init__(self, opt):
        super(MultiSpaceTxtNet_bow_w2v_gru, self).__init__()
        self.txt_net_b = BoWTxtNet(opt)

        self.txt_net_w = W2VTxtNet(opt)

        self.txt_net_g = GruTxtNet(opt)

    def forward(self, txt_input, cap_ids):
        feature_b = self.txt_net_b(txt_input, cap_ids)
        feature_w = self.txt_net_w(txt_input, cap_ids)
        feature_g = self.txt_net_g(txt_input, cap_ids)

        return feature_b, feature_w, feature_g

class MultiSpaceTxtNet_bow_w2v_gru_bert (MultiSpaceTxtNet_bow_w2v_gru):
    def __init__(self, opt):
        super(MultiSpaceTxtNet_bow_w2v_gru_bert, self).__init__(opt)
        self.txt_net_bt = BertTxtNet(opt)

    def forward(self, txt_input, cap_ids):
        feature_b = self.txt_net_b(txt_input, cap_ids)
        feature_w = self.txt_net_w(txt_input, cap_ids)
        feature_g = self.txt_net_g(txt_input, cap_ids)
        feature_bt = self.txt_net_bt(txt_input, cap_ids)


        return feature_b, feature_w, feature_g, feature_bt

class MultiSpaceTxtNet_bow_w2v_lstm (MultiSpaceTxtNet_bow_w2v_gru):
    def __init__(self, opt):
        self.txt_net_b = BoWTxtNet(opt)

        self.txt_net_w = W2VTxtNet(opt)

        self.txt_net_g = LstmTxtNet(opt)


class MultiSpaceTxtNet_bow_w2v_infersent(nn.Module):
    def __init__(self, opt):
        super(MultiSpaceTxtNet_bow_w2v_infersent, self).__init__()
        self.txt_net_b = BoWTxtNet(opt)
        self.txt_net_w = W2VTxtNet(opt)
        self.txt_net_i = InfersentTxtNet(opt)

    def forward(self, txt_input, cap_ids):
        feature_b = self.txt_net_b(txt_input, cap_ids)
        feature_w = self.txt_net_w(txt_input, cap_ids)
        feature_i = self.txt_net_i(txt_input, cap_ids)

        return feature_b, feature_w, feature_i
 
class MultiSpaceTxtNet_softbow_w2v_gru (nn.Module):
    def __init__(self, opt):
        super(MultiSpaceTxtNet_softbow_w2v_gru, self).__init__()
        self.txt_net_b = SoftBoWTxtNet(opt)
        self.txt_net_w = W2VTxtNet(opt)
        self.txt_net_g = GruTxtNet(opt)

    def forward(self, txt_input, cap_ids):
        feature_b = self.txt_net_b(txt_input, cap_ids)
        feature_w = self.txt_net_w(txt_input, cap_ids)
        feature_g = self.txt_net_g(txt_input, cap_ids)

        return feature_b, feature_w, feature_g

class MultiSpaceTxtNet_softbow_w2v_infersent (nn.Module):
    def __init__(self, opt):
        super(MultiSpaceTxtNet_softbow_w2v_infersent, self).__init__()
        self.txt_net_b = SoftBoWTxtNet(opt)
        self.txt_net_w = W2VTxtNet(opt)
        self.txt_net_i = InfersentTxtNet(opt)

    def forward(self, txt_input, cap_ids):
        feature_b = self.txt_net_b(txt_input, cap_ids)
        feature_w = self.txt_net_w(txt_input, cap_ids)
        feature_i = self.txt_net_i(txt_input, cap_ids)

        return feature_b, feature_w, feature_i


class MultiSpaceTxtNet_bow_w2v_bigru (nn.Module):
    def __init__(self, opt):
        super(MultiSpaceTxtNet_bow_w2v_bigru, self).__init__()
        self.txt_net_b = BoWTxtNet(opt)
        self.txt_net_w = W2VTxtNet(opt)
        self.txt_net_g = BiGruTxtNet(opt)

    def forward(self, txt_input, cap_ids):
        feature_b = self.txt_net_b(txt_input, cap_ids)
        feature_w = self.txt_net_w(txt_input, cap_ids)
        feature_g = self.txt_net_g(txt_input, cap_ids)

        return feature_b, feature_w, feature_g

class MultiSpaceTxtNet_bow_w2v_bigru_bert (MultiSpaceTxtNet_bow_w2v_bigru):
    def __init__(self, opt):
        super(MultiSpaceTxtNet_bow_w2v_bigru_bert, self).__init__(opt)
        self.txt_net_bt = BertTxtNet(opt)

    def forward(self, txt_input, cap_ids):
        feature_b = self.txt_net_b(txt_input, cap_ids)
        feature_w = self.txt_net_w(txt_input, cap_ids)
        feature_g = self.txt_net_g(txt_input, cap_ids)
        feature_bt = self.txt_net_bt(txt_input, cap_ids)

        return feature_b, feature_w, feature_g, feature_bt

class MultiSpaceTxtNet_bow_netvlad_bigru (nn.Module):
    def __init__(self, opt):
        super(MultiSpaceTxtNet_bow_netvlad_bigru, self).__init__()
        self.txt_net_b = BoWTxtNet(opt)
        self.txt_net_n = W2V_NetVLADTxtNet(opt)
        self.txt_net_g = BiGruTxtNet(opt)

    def forward(self, txt_input, cap_ids):
        feature_b = self.txt_net_b(txt_input, cap_ids)
        feature_n = self.txt_net_n(txt_input, cap_ids)
        feature_g = self.txt_net_g(txt_input, cap_ids)

        return feature_b, feature_n, feature_g


class MultiSpaceTxtNet_softbow_w2v_bigru (nn.Module):
    def __init__(self, opt):
        super(MultiSpaceTxtNet_softbow_w2v_bigru, self).__init__()

        self.txt_net_b = SoftBoWTxtNet(softbow_opt)

        self.txt_net_w = W2VTxtNet(w2v_opt)

        self.txt_net_g = BiGruTxtNet(opt)

    def forward(self, txt_input, cap_ids):
        feature_b = self.txt_net_b(txt_input, cap_ids)
        feature_w = self.txt_net_w(txt_input, cap_ids)
        feature_g = self.txt_net_g(txt_input, cap_ids)

        return feature_b, feature_w, feature_g


class MultiSpaceTxtNet_softbow_w2v_bigru_infersent (nn.Module):
    def __init__(self, opt):
        super(MultiSpaceTxtNet_softbow_w2v_bigru_infersent, self).__init__()
        self.txt_net_b = SoftBoWTxtNet(opt)

        self.txt_net_w = W2VTxtNet(opt)

        self.txt_net_g = BiGruTxtNet(opt)
        
        self.txt_net_i = InfersentTxtNet(opt)
        

    def forward(self, txt_input, cap_ids):
        feature_b = self.txt_net_b(txt_input, cap_ids)
        feature_w = self.txt_net_w(txt_input, cap_ids)
        feature_g = self.txt_net_g(txt_input, cap_ids)
        feature_i = self.txt_net_i(txt_input, cap_ids)

        return feature_b, feature_w, feature_g ,feature_i
    
    
class MultiSpaceTxtNet_bow_w2v (nn.Module):
    def __init__(self, opt):
        super(MultiSpaceTxtNet_bow_w2v, self).__init__()
        self.txt_net_b = BoWTxtNet(opt)
        self.txt_net_w = W2VTxtNet(opt)

    def forward(self, txt_input, cap_ids):
        feature_b = self.txt_net_b(txt_input, cap_ids)
        feature_w = self.txt_net_w(txt_input, cap_ids)

        return feature_b, feature_w

class MultiSpaceTxtNet_bow_w2v_bert (MultiSpaceTxtNet_bow_w2v):
    def __init__(self, opt):
        super(MultiSpaceTxtNet_bow_w2v_bert, self).__init__(opt)
        self.txt_net_bt = BertTxtNet(opt)

    def forward(self, txt_input, cap_ids):
        feature_b = self.txt_net_b(txt_input, cap_ids)
        feature_w = self.txt_net_w(txt_input, cap_ids)
        feature_bt = self.txt_net_bt(txt_input, cap_ids)

        return feature_b, feature_w, feature_bt


class MultiSpaceTxtNet_bow_netvlad (nn.Module):
    def __init__(self, opt):
        super(MultiSpaceTxtNet_bow_netvlad, self).__init__()

        self.txt_net_b = BoWTxtNet(opt)
        self.txt_net_w = W2V_NetVLADTxtNet(opt)

    def forward(self, txt_input, cap_ids):
        feature_b = self.txt_net_b(txt_input, cap_ids)
        feature_w = self.txt_net_w(txt_input, cap_ids)

        return feature_b, feature_w


class MultiSpaceTxtNet_bow_gru (nn.Module):
    def __init__(self, opt):
        super(MultiSpaceTxtNet_bow_gru, self).__init__()
        self.txt_net_b = BoWTxtNet(opt)
        self.txt_net_g = GruTxtNet(opt)

    def forward(self, txt_input, cap_ids):
        feature_b = self.txt_net_b(txt_input, cap_ids)
        feature_g = self.txt_net_g(txt_input, cap_ids)

        return feature_b, feature_g


class MultiSpaceTxtNet_w2v_gru (nn.Module):
    def __init__(self, opt):
        super(MultiSpaceTxtNet_w2v_gru, self).__init__()
        self.txt_net_w = W2VTxtNet(opt)
        self.txt_net_g = GruTxtNet(opt)

    def forward(self, txt_input, cap_ids):
        feature_w = self.txt_net_w(txt_input, cap_ids)
        feature_g = self.txt_net_g(txt_input, cap_ids)

        return feature_w, feature_g



# Deprecated model
class OneLossMultiSpaceTxtNet (nn.Module):
    def __init__(self, opt):
        super(OneLossMultiSpaceTxtNet, self).__init__()

        opt.txt_fc_layers[0] = opt.t2v_bow.ndims
        self.txt_net_b1 = BoWTxtNet(opt)
        self.txt_net_b2 = BoWTxtNet(opt)

        opt.txt_fc_layers[0] = opt.w2v_out_size
        self.txt_net_w1 = W2VTxtNet(opt)
        self.txt_net_w2 = W2VTxtNet(opt)

        opt.txt_fc_layers[0] = opt.rnn_size*2 if opt.pooling == 'mean_last' else opt.rnn_size
        self.txt_net_g1 = GruTxtNet(opt)
        self.txt_net_g2 = GruTxtNet(opt)

    def forward(self, txt_input, cap_ids):
        feature_b1 = self.txt_net_b1(txt_input)
        feature_b2 = self.txt_net_b2(txt_input)
        feature_w1 = self.txt_net_w1(txt_input)
        feature_w2 = self.txt_net_w2(txt_input)
        feature_g1 = self.txt_net_g1(txt_input)
        feature_g2 = self.txt_net_g2(txt_input)

        return feature_b1, feature_w1, feature_g1, feature_b2, feature_w2, feature_g2


class MultiSpaceVisNet (nn.Module):
    def __init__(self, opt):
        super(MultiSpaceVisNet, self).__init__()
        self.vis_net_b = VisTransformNet(opt)
        self.vis_net_w = VisTransformNet(opt)
        self.vis_net_g = VisTransformNet(opt)
        self.vis_net_m = VisTransformNet(opt)

    def forward(self, vis_input):
        feature_m = self.vis_net_m(vis_input)
        feature_b = self.vis_net_b(vis_input)
        feature_w = self.vis_net_w(vis_input)
        feature_g = self.vis_net_g(vis_input)

        return feature_m, feature_b, feature_w, feature_g



class MultiSpaceVisNet_2 (nn.Module):
    def __init__(self, opt):
        super(MultiSpaceVisNet_2, self).__init__()
        self.vis_net_1 = VisTransformNet(opt)
        self.vis_net_2 = VisTransformNet(opt)

    def forward(self, vis_input):
        feature_1 = self.vis_net_1(vis_input)
        feature_2 = self.vis_net_2(vis_input)

        return feature_1, feature_2


class MultiSpaceVisNet_NetVLAD_2 (nn.Module):
    def __init__(self, opt):
        super(MultiSpaceVisNet_NetVLAD_2, self).__init__()
        self.encoder = NetVLAD(opt)
        self.vis_net_1 = VisTransformNet(opt)
        self.vis_net_2 = VisTransformNet(opt)

    def forward(self, vis_input):
        encoder_output = self.encoder(vis_input)
        feature_1 = self.vis_net_1(encoder_output)
        feature_2 = self.vis_net_2(encoder_output)

        return feature_1, feature_2


class MultiSpaceVisNet_3 (nn.Module):
    def __init__(self, opt):
        super(MultiSpaceVisNet_3, self).__init__()
        self.vis_net_1 = VisTransformNet(opt)
        self.vis_net_2 = VisTransformNet(opt)
        self.vis_net_3 = VisTransformNet(opt)

    def forward(self, vis_input):
        feature_1 = self.vis_net_1(vis_input)
        feature_2 = self.vis_net_2(vis_input)
        feature_3 = self.vis_net_3(vis_input)

        return feature_1, feature_2, feature_3

class MultiSpaceVisNet_4 (nn.Module):
    def __init__(self, opt):
        super(MultiSpaceVisNet_4, self).__init__()
        self.vis_net_1 = VisTransformNet(opt)
        self.vis_net_2 = VisTransformNet(opt)
        self.vis_net_3 = VisTransformNet(opt)
        self.vis_net_4 = VisTransformNet(opt)

    def forward(self, vis_input):
        feature_1 = self.vis_net_1(vis_input)
        feature_2 = self.vis_net_2(vis_input)
        feature_3 = self.vis_net_3(vis_input)
        feature_4 = self.vis_net_3(vis_input)
        
        return feature_1, feature_2, feature_3,feature_4


class CrossModalNetwork(object):

    def _init_vis_net(self, opt):
        self.vis_net = VisNet(opt)

    def _init_txt_net(self, opt):
        self.txt_net = TxtNet(opt)

    def _init_optim(self, opt):
        self.grad_clip = opt.grad_clip
        if torch.cuda.is_available():
            cudnn.benchmark = True

        if opt.optimizer == 'adam':
            self.optimizer = torch.optim.Adam(self.params, lr=opt.lr)
        elif opt.optimizer == 'rmsprop':
            self.optimizer = torch.optim.RMSprop(self.params, lr=opt.lr)

        self.lr_schedulers = [torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=opt.lr_decay_rate),
                      torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', factor=0.5, patience=2)]

        self.iters = 0

    def _init_loss(self, opt):
        if opt.loss == 'mrl':
            self.criterion = MarginRankingLoss(margin=opt.margin,
                                             measure=opt.measure,
                                             max_violation=opt.max_violation,
                                             cost_style=opt.cost_style,
                                             direction=opt.direction)
        elif opt.loss == 'mse':
            self.criterion = nn.MSELoss()

    def __init__(self, opt):
        self._init_vis_net(opt)
        self._init_txt_net(opt)

        if torch.cuda.is_available():
            #self.vis_net.cuda()
            #self.txt_net.cuda()
            #if torch.cuda.device_count() > 1:
            #    self.vis_net = nn.DataParallel(self.vis_net)
            #    self.txt_net = nn.DataParallel(self.txt_net)
            self.vis_net.to(device)
            self.txt_net.to(device)

        self.params = list(self.txt_net.parameters())
        self.params += list(self.vis_net.parameters())

        self._init_loss(opt)
        self._init_optim(opt)

        self.measure = opt.measure
        
        # 
        # print self.vis_net.state_dict()
        # print self.txt_net.state_dict()
 
    def state_dict(self):
        state_dict = [self.vis_net.state_dict(), self.txt_net.state_dict()]
        return state_dict

    def load_state_dict(self, state_dict):
        self.vis_net.load_state_dict(state_dict[0])
        # print 'vis_net_state_dict:', state_dict[0]
        self.txt_net.load_state_dict(state_dict[1])
        # print 'txt_net_state_dict:', state_dict[1]

    def switch_to_train(self):
        self.vis_net.train()
        self.txt_net.train()

    def switch_to_eval(self):
        self.vis_net.eval()
        self.txt_net.eval()

    @property
    def learning_rate(self):
        """Return learning rate"""
        lr_list = []
        for param_group in self.optimizer.param_groups:
            lr_list.append(param_group['lr'])
        return lr_list

    def lr_step(self, val_value):
        self.lr_schedulers[0].step()
        self.lr_schedulers[1].step(val_value)

    def compute_similarity(self, txt_embs, vis_embs):
        return cosine_sim(txt_embs, vis_embs)
        #return -1 * euclidean_dist(txt_embs, vis_embs)

    def compute_loss(self, txt_embs, vis_embs):
        """Compute the loss given pairs of image and caption embeddings
        """
        loss = self.criterion(txt_embs, vis_embs)
        #loss, indices_im = self.criterion(txt_embs, vis_embs)
        return loss, None

    def train(self, txt_input, cap_ids, vis_input, vis_ids):
        """One training step given vis_feats and captions.
        """
        self.iters += 1

        # compute the embeddings
        txt_embs = self.txt_net(txt_input, cap_ids)
        vis_embs = self.vis_net(vis_input)

        # measure accuracy and record loss
        self.optimizer.zero_grad()
        loss, indices_im = self.compute_loss(txt_embs, vis_embs)

        loss_value = loss.item()

        # compute gradient and do SGD step
        loss.backward()
        if self.grad_clip > 0:
            clip_grad_norm_(self.params, self.grad_clip)
        self.optimizer.step()

        return loss_value, indices_im

    def validate_similarity(self, data_loader):
        self.switch_to_eval()

        vis_embs = None
        txt_embs = None

        pbar = Progbar(len(data_loader.dataset))
        

        for i, (vis_input, txt_input, idxs, batch_vis_ids, batch_txt_ids) in enumerate(data_loader):
            with torch.no_grad():
                txt_emb = self.txt_net(txt_input)
                vis_emb = self.vis_net(vis_input)

            if vis_embs is None:
                txt_embs = torch.zeros(len(data_loader.dataset), txt_emb.size(1))
                vis_embs = torch.zeros(len(data_loader.dataset), vis_emb.size(1))

            txt_embs[idxs] = txt_emb.cpu().clone()
            vis_embs[idxs] = vis_emb.cpu().clone()

            pbar.add(len(idxs))

        return self.compute_similarity(txt_embs, vis_embs).numpy()

    @util.timer
    def predict(self, txt_loader, vis_loader):
        self.switch_to_eval()

        txt_ids = []
        vis_ids = []
        pbar = Progbar(len(txt_loader.dataset))

        # total_query2common_spcace_time = 0.0
        # total_vis2common_spcace_time = 0.0
        # total_cal_sim_time = 0.0
        # # total_predict_time = 0.0
        # pre_time = time.time()
        for i, (txt_input, idxs, batch_txt_ids) in enumerate(txt_loader):
            with torch.no_grad():
                # q2s_time = time.time()
                txt_emb = self.txt_net(txt_input, batch_txt_ids)
                # total_query2common_spcace_time += (time.time() - q2s_time)

            for j, (vis_input, idxs, batch_vis_ids) in enumerate(vis_loader):
                # v2s_time = time.time()
                with torch.no_grad():
                    vis_emb = self.vis_net(vis_input)
                # total_vis2common_spcace_time += (time.time() - v2s_time)
                # cs_time = time.time()                
                score = self.compute_similarity(txt_emb, vis_emb)
                # total_cal_sim_time += (time.time() - cs_time)
                if i == 0:
                    vis_ids.extend(batch_vis_ids)
                batch_score = score.cpu() if j==0 else torch.cat((batch_score, score.cpu()), dim=1)

            pbar.add(len(batch_txt_ids))
            txt_ids.extend(batch_txt_ids)
            scores = batch_score if i==0 else torch.cat((scores, batch_score), dim=0)
        # print("total_predict_time: ",time.time() - pre_time)
        # print("total_query2common_spcace_time: ", total_query2common_spcace_time)
        # print('total_vis2common_spcace_time', total_vis2common_spcace_time)
        # print("total_cal_sim_time: ",total_cal_sim_time)

        return scores.numpy(), txt_ids, vis_ids

    def embed_vis(self, vis_input):
        self.switch_to_eval()
        vis_input = np.array(vis_input)
        if vis_input.ndim == 1:
            vis_input = [vis_input]

        with torch.no_grad():
            vis_input = torch.Tensor(vis_input).to(device)
            vis_embs = self.vis_net(vis_input)

        return vis_embs.cpu()

    def embed_txt(self, txt_input):
        self.switch_to_eval()
        if isinstance(txt_input, str):
            txt_input = [txt_input]

        with torch.no_grad():
            txt_embs = self.txt_net(txt_input)

        return txt_embs.cpu()


class MultiSpaceModel(CrossModalNetwork):

    def _init_vis_net(self, opt):
        self.vis_net = MultiSpaceVisNet(opt)

    def _init_txt_net(self, opt):
        self.txt_net = MultiSpaceTxtNet(opt)

    def compute_similarity(self, txt_embs, vis_embs):
        scores = cosine_sim(txt_embs[0], vis_embs[0]) 
        for i in range(1, len(txt_embs)):
            #scores = torch.max(scores, cosine_sim(txt_embs[i], vis_embs[i]))
            scores += cosine_sim(txt_embs[i], vis_embs[i])
        #return scores*1.0/len(txt_embs)
        return scores*1.0

    def compute_loss(self, txt_embs, vis_embs):

        #loss, indices_im = self.criterion(txt_embs[0], vis_embs[0])
        loss = self.criterion(txt_embs[0], vis_embs[0])
        indices_im = None
        for i in range(1, len(vis_embs)):
            #cur_loss, cur_indices_im = self.criterion(txt_embs[i], vis_embs[i])
            cur_loss = self.criterion(txt_embs[i], vis_embs[i])
            loss += cur_loss
            #indices_im = torch.cat([indices_im, cur_indices_im], dim=1)
        return loss, indices_im

    def validate_similarity(self, data_loader):
        self.switch_to_eval()

        vis_embs, txt_embs = None, None

        pbar = Progbar(len(data_loader.dataset))
        for _, (vis_input, txt_input, idxs, batch_vis_ids, batch_txt_ids) in enumerate(data_loader):
            with torch.no_grad():
                txt_emb = self.txt_net(txt_input)
                vis_emb = self.vis_net(vis_input)

            if vis_embs is None:
                txt_embs = [torch.zeros(len(data_loader.dataset), txt_emb[0].size(1)) for i in range(len(txt_emb))]
                vis_embs = [torch.zeros(len(data_loader.dataset), vis_emb[0].size(1)) for i in range(len(vis_emb))]

            for i in range(len(txt_emb)):
                txt_embs[i][idxs] = txt_emb[i].cpu().clone()

            for i in range(len(vis_emb)):
                vis_embs[i][idxs] = vis_emb[i].cpu().clone()

            pbar.add(len(idxs))

        return self.compute_similarity(txt_embs, vis_embs)


class MultiSpaceModel_bow_w2v_gru(MultiSpaceModel):

    def _init_vis_net(self, opt):
        self.vis_net = MultiSpaceVisNet_3(opt)

    def _init_txt_net(self, opt):
        self.txt_net = MultiSpaceTxtNet_bow_w2v_gru(opt)

def MultiSpaceModel_bow_w2v_lstm(MultiSpaceModel_bow_w2v_gru):
    def __init_txt_net(self, opt):
        self.txt_net = MultiSpaceTxtNet_bow_w2v_lstm(opt)


class MultiSpaceModel_bow_w2v_bigru(MultiSpaceModel_bow_w2v_gru):
    def _init_txt_net(self, opt):
        self.txt_net = MultiSpaceTxtNet_bow_w2v_bigru(opt)


class MultiSpaceModel_bow_netvlad_bigru(MultiSpaceModel_bow_w2v_gru):
    def _init_txt_net(self, opt):
        self.txt_net = MultiSpaceTxtNet_bow_netvlad_bigru(opt)

class MultiSpaceModel_softbow_w2v_gru(MultiSpaceModel_bow_w2v_gru):
    def _init_txt_net(self, opt):
        self.txt_net = MultiSpaceTxtNet_softbow_w2v_gru(opt)

class MultiSpaceModel_softbow_w2v_bigru(MultiSpaceModel_bow_w2v_gru):
    def _init_txt_net(self, opt):
        self.txt_net = MultiSpaceTxtNet_softbow_w2v_bigru(opt)


class MultiSpaceModel_bow_w2v_infersent(MultiSpaceModel):

    def _init_vis_net(self, opt):
        self.vis_net = MultiSpaceVisNet_3(opt)

    def _init_txt_net(self, opt):
        self.txt_net = MultiSpaceTxtNet_bow_w2v_infersent(opt)

class MultiSpaceModel_bow_w2v_bert(MultiSpaceModel):

    def _init_vis_net(self, opt):
        self.vis_net = MultiSpaceVisNet_3(opt)

    def _init_txt_net(self, opt):
        self.txt_net = MultiSpaceTxtNet_bow_w2v_bert(opt)


class MultiSpaceModel_softbow_w2v_infersent(MultiSpaceModel_bow_w2v_infersent):

    def _init_vis_net(self, opt):
        self.vis_net = MultiSpaceVisNet_3(opt)

    def _init_txt_net(self, opt):
        self.txt_net = MultiSpaceTxtNet_softbow_w2v_infersent(opt)

class MultiSpaceModel_bow_w2v_gru_bert(MultiSpaceModel):

    def _init_vis_net(self, opt):
        self.vis_net = MultiSpaceVisNet_4(opt)

    def _init_txt_net(self, opt):
        self.txt_net = MultiSpaceTxtNet_bow_w2v_gru_bert(opt)


class MultiSpaceModel_bow_w2v_bigru_bert(MultiSpaceModel):

    def _init_vis_net(self, opt):
        self.vis_net = MultiSpaceVisNet_4(opt)

    def _init_txt_net(self, opt):
        self.txt_net = MultiSpaceTxtNet_bow_w2v_bigru_bert(opt)


class MultiSpaceModel_softbow_w2v_bigru_infersent(MultiSpaceModel):

    def _init_vis_net(self, opt):
        self.vis_net = MultiSpaceVisNet_4(opt)

    def _init_txt_net(self, opt):
        self.txt_net = MultiSpaceTxtNet_softbow_w2v_bigru_infersent(opt)

 
class MultiSpaceModel_bow_w2v(MultiSpaceModel):
    
    def _init_vis_net(self, opt):
        self.vis_net = MultiSpaceVisNet_2(opt)

    def _init_txt_net(self, opt):
        self.txt_net = MultiSpaceTxtNet_bow_w2v(opt)

class MultiSpaceModel_bow_w2v_euclidean(MultiSpaceModel_bow_w2v):
    def compute_similarity(self, txt_embs, vis_embs):
        scores = enclidean_sim(txt_embs[0], vis_embs[0]) 
        for i in range(1, len(txt_embs)):
            #scores = torch.max(scores, cosine_sim(txt_embs[i], vis_embs[i]))
            scores += enclidean_sim(txt_embs[i], vis_embs[i])
        #return scores*1.0/len(txt_embs)
        return scores*1.0

class MultiSpaceModel_bow_w2v_euclideandist(MultiSpaceModel_bow_w2v):
    def compute_similarity(self, txt_embs, vis_embs):
        print("scores euclidean_dist")
        scores = -1.0 *euclidean_dist(txt_embs[0], vis_embs[0]) 
        for i in range(1, len(txt_embs)):
            #scores = torch.max(scores, cosine_sim(txt_embs[i], vis_embs[i]))
            scores += -1.0 * euclidean_dist(txt_embs[i], vis_embs[i])
        #return scores*1.0/len(txt_embs)
        return scores*1.0





class MultiSpaceModel_bow_netvlad(MultiSpaceModel):

    def _init_vis_net(self, opt):
        self.vis_net = MultiSpaceVisNet_2(opt)

    def _init_txt_net(self, opt):
        self.txt_net = MultiSpaceTxtNet_bow_netvlad(opt)


class MultiSpaceModel_visnetvlad_bow_w2v(MultiSpaceModel):

    def _init_vis_net(self, opt):
        self.vis_net = MultiSpaceVisNet_NetVLAD_2(opt)

    def _init_txt_net(self, opt):
        self.txt_net = MultiSpaceTxtNet_bow_w2v(opt)


class MultiSpaceModel_bow_gru(MultiSpaceModel):

    def _init_vis_net(self, opt):
        self.vis_net = MultiSpaceVisNet_2(opt)

    def _init_txt_net(self, opt):
        self.txt_net = MultiSpaceTxtNet_bow_gru(opt)


class MultiSpaceModel_w2v_gru(MultiSpaceModel):

    def _init_vis_net(self, opt):
        self.vis_net = MultiSpaceVisNet_2(opt)

    def _init_txt_net(self, opt):
        self.txt_net = MultiSpaceTxtNet_w2v_gru(opt)


# Deprecated Model
class OneLossMultiSpaceModel_bow_w2v_gru(CrossModalNetwork):

    def _init_vis_net(self, opt):
        self.vis_net = MultiSpaceVisNet_3(opt)

    def _init_txt_net(self, opt):
        self.txt_net = MultiSpaceTxtNet_bow_w2v_gru(opt)

    def _init_loss(self, opt):
        self.criterion = MarginRankingLoss_adv(margin=opt.margin,
                                             max_violation=opt.max_violation,
                                             cost_style=opt.cost_style,
                                             direction=opt.direction)

    def compute_similarity(self, txt_embs, vis_embs):
        # shared common space
        scores = cosine_sim(txt_embs[0], vis_embs[0])
        # distict common space
        for i in range(1, len(txt_embs)):
            scores += cosine_sim(txt_embs[i], vis_embs[i])

        return scores*1.0/len(txt_embs)

    def compute_loss(self, txt_embs, vis_embs):
        scores = self.compute_similarity(txt_embs, vis_embs)

        loss = self.criterion(scores)
        return loss

    def validate_similarity(self, data_loader):
        self.switch_to_eval()

        txt_embs, vis_embs = None, None

        pbar = Progbar(len(data_loader.dataset))
        for _, (vis_input, txt_input, idxs, batch_vis_ids, batch_txt_ids) in enumerate(data_loader):
            with torch.no_grad():
                txt_emb = self.txt_net(txt_input)
                vis_emb = self.vis_net(vis_input)

            if vis_embs is None:
                txt_embs = [torch.zeros(len(data_loader.dataset), txt_emb[0].size(1)) for i in range(len(txt_emb))]
                vis_embs = [torch.zeros(len(data_loader.dataset), vis_emb[0].size(1)) for i in range(len(vis_emb))]

            for i in range(len(txt_emb)):
                txt_embs[i][idxs] = txt_emb[i].cpu().clone()

            for i in range(len(vis_emb)):
                vis_embs[i][idxs] = vis_emb[i].cpu().clone()

            pbar.add(len(idxs))

        return self.compute_similarity(txt_embs, vis_embs)


class OneLossMultiSpaceModel_bow_w2v(OneLossMultiSpaceModel_bow_w2v_gru):
    def _init_vis_net(self, opt):
        self.vis_net = MultiSpaceVisNet_2(opt)

    def _init_txt_net(self, opt):
        self.txt_net = MultiSpaceTxtNet_bow_w2v(opt)


class W2VV (CrossModalNetwork):
    def _init_vis_net(self, opt):
        self.vis_net = IdentityNet(opt)

    def _init_txt_net(self, opt):
        self.txt_net = MultiScaleTxtNet_gru_w2v_bow(opt)


class W2VV_bert_bigru_w2v_bow (W2VV):
    def _init_txt_net(self, opt):
        self.txt_net = MultiScaleTxtNet_bert_bigru_w2v_bow(opt)


class W2VVPP (CrossModalNetwork):
    def _init_vis_net(self, opt):
        self.vis_net = VisTransformNet(opt)
        #self.vis_net = VisNet(opt)
        #self.vis_net = AvgPoolVisNet(opt)

    def _init_txt_net(self, opt):
        self.txt_net = MultiScaleTxtNet_gru_w2v_bow(opt)


class MultiScaleModel_w2v_bow(W2VVPP):
    def _init_txt_net(self, opt):
        self.txt_net = MultiScaleTxtNet_w2v_bow(opt)

class MultiScaleModel_w2v_bow_euclidean(MultiScaleModel_w2v_bow):
    def compute_similarity(self, txt_embs, vis_embs):
        # print 'enclidean_sim(txt_embs, vis_embs)[:10]', enclidean_sim(txt_embs, vis_embs)[:10]
        return enclidean_sim(txt_embs, vis_embs)


class MultiScaleModel_netvlad_bow(W2VVPP):
    def _init_txt_net(self, opt):
        self.txt_net = MultiScaleTxtNet_netvlad_bow(opt)

class MultiScaleModel_bigru_netvlad_bow(W2VVPP):
    def _init_txt_net(self, opt):
        self.txt_net = MultiScaleTxtNet_bigru_netvlad_bow(opt)

class MultiScaleModel_lstm_w2v_bow(W2VVPP):
    def _init_txt_net(self, opt):
        self.txt_net = MultiScaleTxtNet_lstm_w2v_bow(opt)

class MultiScaleModel_gru_w2v_softbow(W2VVPP):
    def _init_txt_net(self, opt):
        self.txt_net = MultiScaleTxtNet_gru_w2v_softbow(opt)

class MultiScaleModel_infersent_w2v_bow(W2VVPP):
    def _init_txt_net(self, opt):
        self.txt_net = MultiScaleTxtNet_infersent_w2v_bow(opt)

class MultiScaleModel_infersent_w2v_softbow(W2VVPP):
    def _init_txt_net(self, opt):
        self.txt_net = MultiScaleTxtNet_infersent_w2v_softbow(opt)

class MultiScaleModel_gruFC_w2vFC_bowFC(W2VVPP):
    def _init_txt_net(self, opt):
        self.txt_net = MultiScaleTxtNet_gruFC_w2vFC_bowFC(opt)

class MultiScaleModel_gru_w2v_bowFC(W2VVPP):
    def _init_txt_net(self, opt):
        self.txt_net = MultiScaleTxtNet_gru_w2v_bowFC(opt)

class MultiScaleModel_bigru_w2v_bow(W2VVPP):
    def _init_txt_net(self, opt):
        self.txt_net = MultiScaleTxtNet_bigru_w2v_bow(opt)

class MultiScaleModel_bert_bigru_w2v_bow(W2VVPP):
    def _init_txt_net(self, opt):
        self.txt_net = MultiScaleTxtNet_bert_bigru_w2v_bow(opt)

class MultiScaleModel_bert_gru_w2v_bow(W2VVPP):
    def _init_txt_net(self, opt):
        self.txt_net = MultiScaleTxtNet_bert_gru_w2v_bow(opt)

class MultiScaleModel_bert_w2v_bow(W2VVPP):
    def _init_txt_net(self, opt):
        self.txt_net = MultiScaleTxtNet_bert_w2v_bow(opt)



class W2VVPP_VLAD (CrossModalNetwork):
    def _init_vis_net(self, opt):
        self.vis_net = NetVLADVisNet(opt)

    def _init_txt_net(self, opt):
        self.txt_net = BoWTxtNet(opt)


class BoWModel (W2VVPP):
    def _init_txt_net(self, opt):
        self.txt_net = BoWTxtNet(opt)

class W2VModel (W2VVPP):
    def _init_txt_net(self, opt):
        self.txt_net = W2VTxtNet(opt)

class W2V_NetVLADModel (W2VVPP):
    def _init_txt_net(self, opt):
        self.txt_net = W2V_NetVLADTxtNet(opt)

class GRUModel (W2VVPP):
    def _init_txt_net(self, opt):
        self.txt_net = GruTxtNet(opt)

class BiGruModel (W2VVPP):
    def _init_txt_net(self, opt):
        self.txt_net = BiGruTxtNet(opt)

class BertModel (W2VVPP):
    def _init_txt_net(self, opt):
        self.txt_net = BertTxtNet(opt)

NAME_TO_MODELS = {'w2vvpp': W2VVPP, 'multispace': MultiSpaceModel, 
                  'bow': BoWModel, 'w2v': W2VModel, 'gru': GRUModel, 'bigru': BiGruModel, 'bert': BertModel,
                #   'w2vv': W2VV, 'w2v_netvlad': W2V_NetVLADModel,
                #   'w2vv_bert_bigru_w2v_bow': W2VV_bert_bigru_w2v_bow,
                  
                #   'multiscale_bigru_netvlad_bow': MultiScaleModel_bigru_netvlad_bow,
                #   'multiscale_gru_w2v_softbow': MultiScaleModel_gru_w2v_softbow,
                  'multiscale_w2v_bow': MultiScaleModel_w2v_bow,
                  'multiscale_gru_w2v_bow': W2VVPP,
                  'multiscale_bigru_w2v_bow':  MultiScaleModel_bigru_w2v_bow,
                  'multiscale_bert_w2v_bow': MultiScaleModel_bert_w2v_bow,
                  'multiscale_bert_gru_w2v_bow': MultiScaleModel_bert_gru_w2v_bow,
                  'multiscale_bert_bigru_w2v_bow': MultiScaleModel_bert_bigru_w2v_bow,
                  
                #   'multiscale_netvlad_bow': MultiScaleModel_netvlad_bow,
                #   'multiscale_lstm_w2v_bow': MultiScaleModel_lstm_w2v_bow,
                #   'multiscale_infersent_w2v_bow': MultiScaleModel_infersent_w2v_bow,
                #   'multiscale_infersent_w2v_softbow': MultiScaleModel_infersent_w2v_softbow,
                #   'multiscale_grufc_w2vfc_bowfc': MultiScaleModel_gruFC_w2vFC_bowFC,
                #   'multiscale_gru_w2v_bowfc': MultiScaleModel_gru_w2v_bowFC,
                                    
                #   'onelossmultispace_bow_w2v_gru': OneLossMultiSpaceModel_bow_w2v_gru, 'w2vvpp_vlad': W2VVPP_VLAD,
                #   'onelossmultispace_bow_w2v': OneLossMultiSpaceModel_bow_w2v,
                  
                  
                #   'multispace_bow_netvlad': MultiSpaceModel_bow_netvlad,
                #   'multispace_bow_netvlad_bigru': MultiSpaceModel_bow_netvlad_bigru,
                #   'multispace_bow_w2v_lstm': MultiSpaceModel_bow_w2v_lstm, 
                #   'multispace_visnetvlad_bow_w2v': MultiSpaceModel_visnetvlad_bow_w2v,
                #   'multispace_bow_gru': MultiSpaceModel_bow_gru, 
                #   'multispace_w2v_gru': MultiSpaceModel_w2v_gru, \

                  'multispace_bow_w2v': MultiSpaceModel_bow_w2v, 
                  'multispace_bow_w2v_gru': MultiSpaceModel_bow_w2v_gru,
                  'multispace_bow_w2v_bigru': MultiSpaceModel_bow_w2v_bigru,
                  'multispace_bow_w2v_bert': MultiSpaceModel_bow_w2v_bert,
                  'multispace_bow_w2v_gru_bert': MultiSpaceModel_bow_w2v_gru_bert,
                  'multispace_bow_w2v_bigru_bert': MultiSpaceModel_bow_w2v_bigru_bert,
                  
                #   'multispace_softbow_w2v_gru': MultiSpaceModel_softbow_w2v_gru,
                #   'multispace_softbow_w2v_bigru': MultiSpaceModel_softbow_w2v_bigru,
                #   'multispace_bow_w2v_infersent': MultiSpaceModel_bow_w2v_infersent,
                #   'multispace_softbow_w2v_infersent': MultiSpaceModel_softbow_w2v_infersent,
                #   'multispace_softbow_w2v_bigru_infersent': MultiSpaceModel_softbow_w2v_bigru_infersent,
      
                #   'multispace_bow_w2v_euclidean': MultiSpaceModel_bow_w2v_euclidean,
                #   'multiscale_bow_w2v_euclidean': MultiScaleModel_w2v_bow_euclidean
                  }

def get_model(name):
    assert name in NAME_TO_MODELS, '%s not supported.'%name
    return NAME_TO_MODELS[name]

if __name__ == '__main__':
    model = get_model('w2vvpp')

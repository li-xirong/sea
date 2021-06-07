import os
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
from aggregator import AvgPooling, MaxPooling, NetVLAD, NetVLAD_AvgPooling
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
    #import pdb; pdb.set_trace()
    #dict(zip(vocab, range(nr_words-1)))

    for i, word in enumerate(renamed):
        idx = vocab.word2idx[word]
        #idx = word_to_idx[word] 
        we[idx] = vecs[i]

    return torch.Tensor(we)


def l2norm(X):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=1, keepdim=True).sqrt()
    X = torch.div(X, norm + 1e-10)
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


class SentFeatBase():
    def __init__(self, collection_sent_feat_dirs:list, feat_name:str) -> None:
        """
        """
        self.feat_name = feat_name
        self.feat_file_list = []
        for d in collection_sent_feat_dirs:
            feat_path = os.path.join(d, self.feat_name)
            feat_file = BigFile(feat_path)
            self.feat_file_list.append(feat_file)
        
        self.names = []
        for f in self.feat_file_list:
            self.names += f.names
        self.names = set(self.names)

    def read(self, cap_ids):
        feat_file = None
        for f in self.feat_file_list:
            if cap_ids[0] in f.names:
                feat_file = f
                break
        assert (feat_file != None)
        feature = [f.read_one(cap_id) for cap_id in cap_ids]
        
        return feature


class IdentityNet(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.output_size = opt.vis_fc_layers[0]

    def forward(self, input_x):
        """Extract image feature vectors."""
        return input_x.to(device)

    def get_output_size(self):
        return self.output_size
       


class TransformNet(nn.Module):
    def __init__(self, opt):
        super().__init__()

        self.fc1 = nn.Linear(opt.fc_layers[0], opt.fc_layers[1])
        if opt.batch_norm:
            self.bn1 = nn.BatchNorm1d(opt. fc_layers[1])
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
        """Copy parameters. overwritting the default one to
        accept state_dict from Full model
        """
        own_state = self.state_dict()
        new_state = OrderedDict()
        for name, param in state_dict.items():
            if name in own_state:
                new_state[name] = param

        super().load_state_dict(new_state)

# Visual models

'''
class VisTransformNet(TransformNet):
    def __init__(self, opt):
        super().__init__(opt.vis_fc_layers, opt)
'''


'''
VisNet conceptually consists of two modules,
+ an encoder which encodes a given visual input into a feature vector
+ a transformer that projects the feature vector into a common space for cross-modal matching
'''
class VisNet(nn.Module):
    def _init_encoder(self, opt):
        self.encoder = IdentityNet(opt)

    def _init_transformer(self, opt):
        self.transformer = TransformNet(opt)

    def get_output_size(self, opt):
        return opt.vis_fc_layers[1]

    def __init__(self, opt):
        super().__init__()
        self._init_encoder(opt)
        # adjust the input size of the transformer 
        opt2 = copy.copy(opt)
        opt2.fc_layers = [self.encoder.get_output_size(), self.get_output_size(opt)]
        # opt2.fc_layers[0] = self.encoder.get_output_size()
        # opt2.fc_lay
        # import pdb; pdb.set_trace()
        self._init_transformer(opt2)

    def forward(self, vis_input):
        features = self.encoder(vis_input)
        features = self.transformer(features)
        return features

    #def load_state_dict(self, state_dict):
    #    super().load_state_dict(state_dict)


class AvgPoolVisNet(VisNet):
    def _init_encoder(self, opt):
        self.encoder = AvgPooling()


class NetVLADVisNet(VisNet):
    def _init_encoder(self, opt):
        self.encoder = NetVLAD_AvgPooling(opt)

# Text models

#class TxtTransformNet(TransformNet):
#    def __init__(self, opt):
#        super().__init__(opt.txt_fc_layers, opt)


class TxtEncoder(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.l2norm = False

    def forward(self, txt_input):
        captions, cap_ids = txt_input
        raise Exception("Not implemented")

    def get_output_size(self):
        return self.output_size


class BoWTxtEncoder(TxtEncoder):
    def __init__(self, opt):
        super().__init__(opt)
        self.t2v = opt.t2v_bow
        self.l2norm = opt.bow_encoder_l2norm if hasattr(opt, 'bow_encoder_l2norm') else False
        self.output_size = self.t2v.ndims
        
    def forward(self, txt_input):
        captions, cap_ids = txt_input  

        out = torch.Tensor([self.t2v.encoding(caption) for caption in captions]).to(device)
        if self.l2norm:
            out = l2norm(out)
        return out


class W2VTxtEncoder(BoWTxtEncoder):
    def __init__(self, opt):
        super().__init__(opt)
        self.t2v = opt.t2v_w2v
        self.l2norm = opt.w2v_encoder_l2norm if hasattr(opt, 'w2v_encoder_l2norm') else False
        self.output_size = opt.t2v_w2v.ndims
    
        

class GruTxtEncoder(TxtEncoder):
    def _init_rnn(self, opt):
        self.rnn = nn.GRU(opt.we_dim,
                          opt.rnn_size,
                          opt.rnn_layer,
                          batch_first=True,
                          dropout=opt.rnn_dropout,
                          bidirectional=False)
        
        self.output_size = opt.rnn_size

    def __init__(self, opt):
        super().__init__(opt)
        self.pooling = opt.pooling
        self.t2v_idx = opt.t2v_idx
        self.we = nn.Embedding(len(self.t2v_idx.vocab), opt.we_dim)
        if opt.we_dim == 500:
            self.we.weight = nn.Parameter(opt.we)  # initialize with a pre-trained 500-dim w2v
        
        self._init_rnn(opt)
        # self.rnn_size = opt.rnn_size
        

    def forward(self, txt_input):
        captions, cap_ids = txt_input
        """Handles variable size captions
        """
        
        batch_size = len(captions)

        # caption encoding
        idx_vecs = [self.t2v_idx.encoding(caption) for caption in captions]
        lengths = [len(vec) for vec in idx_vecs]

        x = torch.zeros(batch_size, max(lengths)).long().to(device)
        for i, vec in enumerate(idx_vecs):
            end = lengths[i]
            x[i, :end] = torch.Tensor(vec)

        # caption embedding
        x = self.we(x)
        packed = pack_padded_sequence(x,
                                      lengths,
                                      batch_first=True,
                                      enforce_sorted=False)

        # Forward propagate RNN
        out, _ = self.rnn(packed)
        # Reshape *final* output to (batch_size, hidden_size)
        padded = pad_packed_sequence(out, batch_first=True)

        if self.pooling == 'mean':
            out = torch.zeros(batch_size, self.output_size).to(device)
            for i, ln in enumerate(lengths):
                out[i] = torch.mean(padded[0][i][:ln], dim=0)
        elif self.pooling == 'last':
            I = torch.LongTensor(lengths).view(-1, 1, 1)
            I = I.expand(batch_size, 1, self.output_size) - 1
            I = I.to(device)
            out = torch.gather(padded[0], 1, I).squeeze(1)
        elif self.pooling == 'mean_last':
            out1 = torch.zeros(batch_size, self.output_size).to(device)
            for i, ln in enumerate(lengths):
                out1[i] = torch.mean(padded[0][i][:ln], dim=0)

            I = torch.LongTensor(lengths).view(-1, 1, 1)
            I = I.expand(batch_size, 1, self.output_size) - 1
            I = I.to(device)
            out2 = torch.gather(padded[0], 1, I).squeeze(1)
            out = torch.cat((out1, out2), dim=1)
        return out


class LstmTxtEncoder(GruTxtEncoder):
    def _init_rnn(self, opt):
        self.rnn = nn.LSTM(opt.we_dim,
                           opt.rnn_size,
                           opt.rnn_layer,
                           batch_first=True,
                           dropout=opt.rnn_dropout,
                           bidirectional=False)
        self.output_size = opt.rnn_size * 2


class BiGruTxtEncoder(GruTxtEncoder):
    def _init_rnn(self, opt):
        self.rnn = nn.GRU(opt.we_dim,
                          opt.rnn_size,
                          opt.rnn_layer,
                          batch_first=True,
                          dropout=opt.rnn_dropout,
                          bidirectional=True)
        self.output_size = opt.rnn_size * 2

    # def __init__(self, opt):
    #     super().__init__(opt)
        # self.rnn_size = opt.rnn_size * 2




class NetVLADTxtEncoder(TxtEncoder):
    def __init__(self, opt):
        super().__init__(opt)
        self.t2v = opt.t2v_w2v
        self.netvlad = NetVLAD(opt)
        self.output_size = opt.netvlad_num_clusters * self.t2v.ndims
        self.l2norm = opt.netvlad_encoder_l2norm if hasattr(opt, 'netvlad_encoder_l2norm') else False
        

    def forward(self, txt_input):
        captions, cap_ids = txt_input      
        out = [torch.Tensor(self.t2v.raw_encoding(caption)) for caption in captions]
        out = self.netvlad(out)
        if self.l2norm:
            out = l2norm(out)
        return out


class BertTxtEncoder(TxtEncoder):
    def __init__(self, opt):
        super().__init__(opt)
        self.has_precomputed, self.is_online = False, False

        if 'bert_online' in opt.text_encoding:
            '''
            refer to 'bert-as-service' library
            '''
            from bert_serving.client import BertClient
            self.bc = BertClient(ip=opt.bert_service_ip, 
                                 port=opt.bert_port, 
                                 port_out=opt.bert_port_out,
                                 check_version=False)
            self.is_online = True
            logger.info('Use BertClient')
        elif 'bert_precomputed' in opt.text_encoding:
            self.bert_feat_base = opt.bert_feat_base
            self.has_precomputed = True
            logger.info('Use precomputed bert features')
        assert (self.is_online or self.has_precomputed) 

        self.output_size = opt.bert_out_size
        self.l2norm = opt.bert_encoder_l2norm if hasattr(opt, 'bert_encoder_l2norm') else False
        

    def forward(self, txt_input):
        captions, cap_ids = txt_input
        assert( cap_ids[0] in self.bert_feat_base.names)
 
        # We prefer to use the precomputed features for efficiency
        if self.has_precomputed:
            bert_feature = self.bert_feat_base.read(cap_ids)
            # bert_feature = [self.txt_bert_feat_file.read_one(cap_id) for cap_id in cap_ids]
            out = torch.Tensor(bert_feature).to(device)
        else:
            out = torch.Tensor(self.bc.encode(captions)).to(device)

        if self.l2norm:
            out = l2norm(out)
        return out


class TxtNet(VisNet):
    def _init_encoder(self, opt):
        self.encoder = TxtEncoder(opt)
    
    def get_output_size(self, opt):
        return opt.txt_fc_layers[1]


class BoWTxtNet(TxtNet):
    def _init_encoder(self, opt):
        self.encoder = BoWTxtEncoder(opt)

class W2VTxtNet(TxtNet):
    def _init_encoder(self, opt):
        self.encoder = W2VTxtEncoder(opt)

class NetVLADTxtNet(TxtNet):
    def _init_encoder(self, opt):
        self.encoder = NetVLADTxtEncoder(opt)
 
# class InfersentTxtNet(TxtNet):
#     def _init_encoder(self, opt):
#         opt.txt_fc_layers[0] = opt.t2v_infer.ndims
#         self.encoder = InfersentTxtEncoder(opt)


class GruTxtNet(TxtNet):
    def _init_encoder(self, opt):
        self.encoder = GruTxtEncoder(opt)


class LstmTxtNet(TxtNet):
    def _init_encoder(self, opt):
        self.encoder = LstmTxtEncoder(opt)


class GruTxtNet_mean_last(GruTxtNet):
    def _init_encoder(self, opt):
        assert opt.pooling == 'mean_last', 'Gru pooling type(%s) not mathed.' % opt.pooling
        self.encoder = GruTxtEncoder(opt)


class BiGruTxtNet(GruTxtNet):
    def _init_encoder(self, opt):
        self.encoder = BiGruTxtEncoder(opt)


class BertTxtNet(TxtNet):
    def _init_encoder(self, opt):
        self.encoder = BertTxtEncoder(opt)


NAME_TO_TXT_NET = {
    'bow': BoWTxtNet,
    'w2v': W2VTxtNet,
    'gru': GruTxtNet,
    'bigru': BiGruTxtNet,
    'bert': BertTxtNet,
    'lstm': LstmTxtNet,
    'netvlad': NetVLADTxtNet
}


class MultiSpaceTxtNet(nn.Module):
    def __init__(self, opt):
        """
        opt includes a list of text nets: opt.txt_net_list
        """
        super().__init__()
        # opt.txt_net_list
        self.txt_nets = []
        for name in opt.txt_net_list:
            txt_net = NAME_TO_TXT_NET[name](opt)
            self.txt_nets.append(txt_net)
        self.txt_nets = nn.ModuleList(self.txt_nets)

    def forward(self, txt_input):  
        features = []
        for txt_net in self.txt_nets:
            feature = txt_net(txt_input)
            features.append(feature)
        
        return tuple(features)


class MultiSpaceVisNet(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.num_of_spaces = len(opt.txt_net_list)
        self.vis_nets = [VisNet(opt) for x in range(self.num_of_spaces)]
        self.vis_nets = nn.ModuleList(self.vis_nets)

    def forward(self, vis_input):
        features = []

        for vis_net in self.vis_nets:
            feature = vis_net(vis_input)
            features.append(feature)
        
        return tuple(features)
        

class CrossModalNet(object):
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

        self.lr_schedulers = [
            torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=opt.lr_decay_rate),
            torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', factor=0.5, patience=2)]

        self.iters = 0

    def _init_loss(self, opt):
        self.criterion = MarginRankingLoss(margin=opt.margin,
                                               similarity=opt.similarity,
                                               max_violation=opt.max_violation,
                                               cost_style=opt.cost_style,
                                               direction=opt.direction)
  
    def __init__(self, opt):
        self._init_vis_net(opt)
        self._init_txt_net(opt)

        if torch.cuda.is_available():
            self.vis_net.to(device)
            self.txt_net.to(device)

        self.params = list(self.txt_net.parameters())
        self.params += list(self.vis_net.parameters())

        self._init_loss(opt)
        self._init_optim(opt)

        self.similarity = opt.similarity

    def state_dict(self):
        state_dict = [self.vis_net.state_dict(), self.txt_net.state_dict()]
        return state_dict

    def load_state_dict(self, state_dict):
        self.vis_net.load_state_dict(state_dict[0])
        self.txt_net.load_state_dict(state_dict[1])

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


    def compute_loss(self, vis_embs, txt_embs):
        """Compute the loss given pairs of image and caption embeddings
        """
        loss = self.criterion(txt_embs, vis_embs)
        return loss, None


    def train(self, captions, cap_ids, vis_feats, vis_ids):
        """One training step given vis_feats and captions.
        """
        self.iters += 1

        # compute the embeddings
        txt_input = (captions, cap_ids)

        txt_embs = self.txt_net(txt_input)
        vis_embs = self.vis_net(vis_feats)

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

    def embed_vis(self, vis_input):
        self.switch_to_eval()
        vis_input = np.array(vis_input)
        if vis_input.ndim == 1:
            vis_input = [vis_input]

        with torch.no_grad():
            vis_input = torch.Tensor(vis_input).to(device)
            vis_embs = self.vis_net(vis_input)

        return vis_embs.cpu()

    # txt_input = tuple([captions, cap_ids])
    def embed_txt(self, txt_input:tuple):
        self.switch_to_eval()
        
        with torch.no_grad():
            txt_embs = self.txt_net(txt_input)

        return txt_embs.cpu()


    def compute_similarity(self, txt_embs, vis_embs):
        return cosine_sim(txt_embs, vis_embs)


    @util.timer
    def predict(self, txt_loader, vis_loader):
        self.switch_to_eval()

        txt_ids = []
        vis_ids = []
        pbar = Progbar(len(txt_loader.dataset))

        for i, (captions, _, cap_ids) in enumerate(txt_loader):
            with torch.no_grad():
                txt_input = tuple([captions, cap_ids])
                txt_embs = self.txt_net(txt_input)

            for j, (vis_input, idxs, batch_vis_ids) in enumerate(vis_loader):

                with torch.no_grad():
                    vis_embs = self.vis_net(vis_input)
                mini_batch_scores = self.compute_similarity(txt_embs, vis_embs)

                if 0 == j:
                    batch_score = mini_batch_scores.cpu()
                else:
                    batch_score = torch.cat((batch_score, mini_batch_scores.cpu()), dim=1)

                if 0 == i:
                    vis_ids.extend(batch_vis_ids)


            pbar.add(len(cap_ids))
            txt_ids.extend(cap_ids)

            if 0 == i:
                scores = batch_score
            else:
                scores = torch.cat((scores, batch_score), dim=0)

        return scores.numpy(), txt_ids, vis_ids


class NaiveCMNet (CrossModalNet):
    def _init_txt_net(self, opt):
        self.txt_net = BoWTxtNet(opt)


def compute_multi_space_similarity(txt_embs:tuple, vis_embs:tuple):
    assert len(txt_embs) == len(vis_embs) 

    scores = cosine_sim(txt_embs[0], vis_embs[0])
    for i in range(1, len(txt_embs)):
        scores += cosine_sim(txt_embs[i], vis_embs[i])
        
    return scores


class SEA (CrossModalNet):
    def _init_vis_net(self, opt):
        self.vis_net = MultiSpaceVisNet(opt)

    def _init_txt_net(self, opt):
        self.txt_net = MultiSpaceTxtNet(opt)

        
    def compute_loss(self, txt_embs, vis_embs):
        """Compute the loss given pairs of image and caption embeddings
        """
        loss = self.criterion(txt_embs[0], vis_embs[0])
        indices_im = None
        for i in range(1, len(vis_embs)):
            cur_loss = self.criterion(txt_embs[i], vis_embs[i])
            loss += cur_loss
        return loss, indices_im


    def compute_similarity(self, txt_embs:tuple, vis_embs:tuple):
        return compute_multi_space_similarity(txt_embs, vis_embs)

 

NAME_TO_MODEL = {
    'naive': NaiveCMNet,
    'sea': SEA
}


def get_model(name):
    assert name in NAME_TO_MODEL, '%s not supported.' % name
    return NAME_TO_MODEL[name]


if __name__ == '__main__':
    model = get_model('naive')
    model = get_model('sea')

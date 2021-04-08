import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AggBaseModel(nn.Module):
    pass

class AvgPooling(AggBaseModel):
    def __init__(self):
        super(AvgPooling, self).__init__()

    def forward(self, x):
        """Handles variable size video frames' feature
        """
        lengths = torch.Tensor([a.size(0) for a in x]).unsqueeze(1).to(device)
        pad_x = pad_sequence(x, batch_first=True).to(device)

        return pad_x.sum(dim=1).div(lengths)


def MaxPooling(AggBaseModel):
    def __init__(self):
        super(MaxPooling, self).__init__()
        super(MaxPooling, self).__init__()

    def forward(self, x):
        """Handles variable size video frames' feature
        """
        pad_x = pad_sequence(x, batch_first=True, padding_value=-np.inf).to(device)
        return pad_x.max(dim=1)


class NetVLAD(AggBaseModel):
    def __init__(self, opt):
        super(NetVLAD, self).__init__()
        self.num_clusters = 32
        self.dim = opt.feature_dim
        self.alpha = 100
        print('num_clusters:', self.num_clusters)
        print('alpha:', self.alpha)

        init_sc = (1 / math.sqrt(opt.feature_dim))
        self.fc1 = nn.Linear(self.dim, self.num_clusters, bias=False)
        #self.centeroids = nn.Parameter(torch.rand(self.num_clusters, self.dim))
        self.centeroids = nn.Parameter(init_sc * torch.randn(self.num_clusters, self.dim))
        self.fc1.weight = nn.Parameter(init_sc * torch.randn(self.num_clusters, self.dim))

        #self._init_params()

    def _init_params(self):
        pass
        #self.fc1.weight = nn.Parameter(2.0 * self.alpha * self.centeroids)
        #self.fc1.bias = nn.Parameter(- self.alpha * self.centeroids.norm(dim=1))

    def forward(self, x):
        """Handles variable size video frames' feature
        params: x: list of _M*D features, 1 <= _M <= M
        """
        vlad = []
        for x_i in x:
            x_i = x_i.to(device)
            M, D = x_i.size()

            x_i = F.normalize(x_i, p=2, dim=-1)  # across descriptor dim

            # soft-assignment
            soft_assign = self.fc1(x_i)  # M*K
            soft_assign = F.softmax(soft_assign, dim=-1)  # M*K

            # M*K*D
            residual = x_i.expand(self.num_clusters, -1, -1).permute(1,0,2) - \
                    self.centeroids.expand(M, -1, -1)   # M*K*D
            residual *= soft_assign.unsqueeze(-1)

            vlad.append(residual.sum(dim=0))

        # N*K*D
        vlad = torch.stack(vlad, 0)

        vlad = F.normalize(vlad, p=2, dim=2)  # intra-normalization
        vlad = vlad.view(vlad.size(0), -1)  # flatten
        vlad = F.normalize(vlad, p=2, dim=1)  # L2 normalize

        return vlad


class NetVLAD_AvgPooling(NetVLAD):
    def __init__(self, opt):
        super(NetVLAD_AvgPooling, self).__init__(opt)
        self.normalize_pooling = False

    def forward(self, x):
        """Handles variable size video frames' feature
        params: x: list of _M*D features, 1 <= _M <= M
        """
        vlad = []
        avgpooling = []
        for x_i in x:
            x_i = x_i.to(device)
            M, D = x_i.size()  # M: # of frames, D: feature dimension

            # mean pooling
            avgpooling.append(x_i.mean(dim=0))

            # normalize input for netvlad
            x_i = F.normalize(x_i, p=2, dim=-1)

            # soft-assignment
            soft_assign = self.fc1(x_i)  # M*K, K: # of center
            soft_assign = F.softmax(soft_assign, dim=-1)  # M*K

            # M*K*D
            residual = x_i.expand(self.num_clusters, -1, -1).permute(1,0,2) - \
                    self.centeroids.expand(M, -1, -1)   # M*K*D
            residual *= soft_assign.unsqueeze(-1)

            vlad.append(residual.sum(dim=0))

        # N*K*D
        vlad = torch.stack(vlad, 0)

        vlad = F.normalize(vlad, p=2, dim=2)  # intra-normalization
        vlad = vlad.view(vlad.size(0), -1)  # flatten
        vlad = F.normalize(vlad, p=2, dim=1)  # L2 normalize

        avgpooling = torch.stack(avgpooling, 0)

        if self.normalize_pooling:
            avgpooling = F.normalize(avgpooling, p=2, dim=1)

        out = torch.cat([avgpooling, vlad], 1)

        return out

def test():
    net = nn.Linear(1024, 64)
    net = net.cuda()
    data1 = torch.randn(2, 1024).cuda()
    data2 = torch.randn(3, 1024).cuda()

    pad_x = pad_sequence([data1, data2], batch_first=True)
    N, M, D = pad_x.shape

    #out = net(pad_x)[0] #pad_x.view(-1, D))
    out = net(pad_x.view(-1, D)).view(N, M, -1)[0]
    print(out.shape)
    out2 = net(data1)
    print(out2.shape)

    print(out[:2])
    print(out2)
    print(out[:2] == out2[:2])
    print(torch.abs(out[:2]-out2[:2]) < 1e-18).sum()



if __name__ == '__main__':
    net = NetVLAD2()
    print(net)

    for x in net.parameters():
        print(x.shape)


import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def l2norm(X):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=1, keepdim=True).sqrt()
    X = torch.div(X, norm)
    return X

def cosine_sim(query, retrio):
    """Cosine similarity between all the query and retrio pairs
    """
    query, retrio = l2norm(query), l2norm(retrio)
    return query.mm(retrio.t())

def euclidean_dist(x, y):
    """Euclidean distance
    https://blog.csdn.net/IT_forlearn/article/details/100022244
    """
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(1, -2, x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()
    return dist


class MarginRankingLoss(nn.Module):
    """
    Compute margin ranking loss
    """
    def __init__(self, margin=0, similarity='cosine', max_violation=False, cost_style='sum', direction='bidir'):
        super(MarginRankingLoss, self).__init__()
        self.margin = margin
        self.cost_style = cost_style
        self.direction = direction
        if similarity == 'cosine':
            self.sim = cosine_sim
        else:
            raise Exception('Similarity %s not implemented.' % similarity)

        self.max_violation = max_violation

    def forward(self, s, im):
        # compute image-sentence score matrix
        scores = self.sim(im, s)
        diagonal = scores.diag().view(im.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        # clear diagonals
        I = torch.eye(scores.size(0)) > .5
        if torch.cuda.is_available():
            I = I.cuda()

        cost_s = None
        cost_im = None
        indices_s = None
        indices_im = None
        # compare every diagonal score to scores in its column
        if self.direction in  ['i2t', 'bidir']:
            # caption retrieval
            cost_s = (self.margin + scores - d1).clamp(min=0)
            cost_s = cost_s.masked_fill_(I, 0)
        # compare every diagonal score to scores in its row
        if self.direction in ['t2i', 'bidir']:
            # image retrieval
            cost_im = (self.margin + scores - d2).clamp(min=0)
            cost_im = cost_im.masked_fill_(I, 0)

        # keep the maximum violating negative for each query
        if self.max_violation:
            if cost_s is not None:
                cost_s, indices_s = cost_s.max(1)
                #cost_s = cost_s.topk(3, 1)[0][:,2]
                #cost_s = cost_s.topk(3, 1)[0].mean(1)
            if cost_im is not None:
                cost_im, indices_im = cost_im.max(0)
                #cost_im = cost_im.topk(3,0)[0][2,:]
                #cost_im = cost_im.topk(3,0)[0].mean(0)

        if cost_s is None:
            cost_s = torch.zeros(1).to(device)
        if cost_im is None:
            cost_im = torch.zeros(1).to(device)

        if self.cost_style == 'sum':
            #return cost_s.sum() + cost_im.sum(), indices_im.unsqueeze(1)
            return cost_s.sum() + cost_im.sum()
        else:
            #return cost_s.mean() + cost_im.mean(), indices_im.unsqueeze(1)
            return cost_s.mean() + cost_im.mean()


class MarginRankingLoss_adv(nn.Module):
    """
    Advanced margin ranking loss
    """
    def __init__(self, margin=0, max_violation=False, cost_style='sum', direction='t2i'):
        super(MarginRankingLoss_adv, self).__init__()
        self.margin = margin
        self.cost_style = cost_style
        self.direction = direction
        self.max_violation = max_violation

    def forward(self, scores):
        """
        scores: t2i matrix
        """
        diagonal = scores.diag().view(scores.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        # clear diagonals
        I = torch.eye(scores.size(0)) > .5
        if torch.cuda.is_available():
            I = I.cuda()

        cost_s = None
        cost_im = None
        indices_s = None
        indices_im = None
        # compare every diagonal score to scores in its column
        if self.direction in  ['t2i', 'bidir']:
            # image retrieval
            cost_im = (self.margin + scores - d1).clamp(min=0)
            cost_im = cost_im.masked_fill_(I, 0)
        # compare every diagonal score to scores in its row
        if self.direction in ['i2t', 'bidir']:
            # caption retrieval
            cost_s = (self.margin + scores - d2).clamp(min=0)
            cost_s = cost_s.masked_fill_(I, 0)

        # keep the maximum violating negative for each query
        if self.max_violation:
            if cost_s is not None:
                cost_s, indices_s = cost_s.max(0)
                #cost_s = cost_s.topk(3, 0)[0][2,:]
                #cost_s = cost_s.topk(3, 0)[0].mean(0)
            if cost_im is not None:
                cost_im, indices_im = cost_im.max(1)
                #cost_im = cost_im.topk(3, 1)[0][:,2]
                #cost_im = cost_im.topk(3, 1)[0].mean(1)

        if cost_s is None:
            cost_s = torch.zeros(1).to(device)
        if cost_im is None:
            cost_im = torch.zeros(1).to(device)

        if self.cost_style == 'sum':
            return cost_s.sum() + cost_im.sum(), indices_im.unsqueeze(1)
        else:
            return cost_s.mean() + cost_im.mean(), indices_im.unsqueeze(1)



if __name__ == '__main__':
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0], [2.0, 5.0, 7.0, 9.0]])
    y = torch.tensor([[3.0, 1.0, 2.0, 5.0], [2.0, 3.0, 4.0, 6.0]])
    dist_matrix = euclidean_dist(x, y)
    print(dist_matrix)

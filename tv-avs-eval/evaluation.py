import numpy as np

def t2i(pred_file):
    ranks = []
    for line in open(pred_file):
        items = line.strip().split()
        assert len(items)%2 == 1
        qry_id = items[0]
        ret_ids = items[1::2]
        rank = ret_ids.index(qry_id.split('#')[0])
        ranks.append(rank)

    # Compute metrics
    ranks = np.array(ranks)
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    mir =np.mean(1./(ranks+1))

    return map(float, [r1, r5, r10, medr, meanr, mir])
    
def process():
    #pred_file = '/home/xcx/VisualSearch/msrvtt10ktest/SimilarityIndex/msrvtt10ktest.caption.txt/msrvtt10ktrain/msrvtt10kval/w2vvpp_resnext101-resnet152_subspace/runs_0/id.sent.score.txt'
    pred_file = '/home/xcx/VisualSearch/msrvtt10ktest/SimilarityIndex/msrvtt10ktest.caption.txt/msrvtt10ktrain/msrvtt10kval/w2vvpp_resnet152_subspace/runs_0/id.sent.score.txt'

    print(t2i(pred_file))


if __name__ == '__main__':
    process()

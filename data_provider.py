import torch
import torch.utils.data as data
import random
import pickle
from bigfile import BigFile
from textlib import TextTool, Vocabulary
from txt2vec import get_lang, BowVec, BowVecNSW, W2Vec, W2VecNSW, IndexVec


def collate_vision(data):
    vis_feats, idxs, vis_ids = zip(*data)
    vis_feats = torch.stack(vis_feats, 0)
    return vis_feats, idxs, vis_ids

def collate_frames(data):
    frame_feats, idxs, vis_ids = zip(*data)
    return frame_feats, idxs, vis_ids

def collate_text(data):
    #data.sort(key=lambda x: len(TextTool.tokenize(x[0])), reverse=True)
    captions, idxs, cap_ids = zip(*data)
    return captions, idxs, cap_ids

# def collate_text_with_cap_feat(data):
#     captions, idxs, cap_ids, cap_features = zip(*data)
#     return captions, idxs, cap_ids, cap_features

def collate_pair(data):
    data.sort(key=lambda x: len(TextTool.tokenize(x[1])), reverse=True)
    vis_feats, captions, idxs, vis_ids, cap_ids = zip(*data)
    vis_feats = torch.stack(vis_feats, 0)
    return vis_feats, captions, idxs, vis_ids, cap_ids

# def collate_pair_with_cap_feat(data):
#     data.sort(key=lambda x: len(TextTool.tokenize(x[1])), reverse=True)
#     vis_feats, captions, idxs, vis_ids, cap_ids, cap_features = zip(*data)
#     vis_feats = torch.stack(vis_feats, 0)
#     return vis_feats, captions, idxs, vis_ids, cap_ids, cap_features  

def collate_frame_pair(data):
    data.sort(key=lambda x: len(TextTool.tokenize(x[1])), reverse=True)
    vis_feats, captions, idxs, vis_ids, cap_ids = zip(*data)
    return vis_feats, captions, idxs, vis_ids, cap_ids


class VisionDataset(data.Dataset):

    def __init__(self, params):
        self.vis_feat_file = BigFile(params['vis_feat']) if isinstance(params['vis_feat'], str) else params['vis_feat']
        self.vis_ids = self.vis_feat_file.names if params.get('vis_ids', None) is None else params['vis_ids']
        self.length = len(self.vis_ids)

    def __getitem__(self, index):
        vis_id = self.vis_ids[index]
        vis_tensor = self.get_feat_by_id(vis_id)
        return vis_tensor, index, vis_id

    def get_feat_by_id(self, vis_id):
        vis_tensor = torch.Tensor(self.vis_feat_file.read_one(vis_id))
        return vis_tensor

    def __len__(self):
        return self.length


class FrameDataset(data.Dataset):

    def __init__(self, params):
        self.vis_feat_file = BigFile(params['vis_feat']) if isinstance(params['vis_feat'], str) else params['vis_feat']
        self.vis_ids = list(set([name.rsplit('_', 1)[0] for name in self.vis_feat_file.names])) if not params.get('vis_ids', None) else params['vis_ids']
        self.length = len(self.vis_ids)
        self.video2frames = {}
        for name in self.vis_feat_file.names:
            self.video2frames.setdefault(name.rsplit('_', 1)[0], []).append(name)
        for name in self.video2frames:
            self.video2frames[name] = sorted(self.video2frames[name], key=lambda x: int(x.rsplit('_',1)[1]))

    def __getitem__(self, index):
        vis_id = self.vis_ids[index]
        vis_tensor = self.get_feat_by_id(vis_id)
        return vis_tensor, index, vis_id

    def get_feat_by_id(self, vis_id):
        frame_ids = self.video2frames[vis_id]
        vis_tensor = torch.Tensor(self.vis_feat_file.read(frame_ids)[1])
        return vis_tensor

    def __len__(self):
        return self.length


class TextDataset(data.Dataset):

    def __init__(self, params):
        capfile = params['capfile']
        self.mask = params.get('caption_mask', False)
        self.captions = {}
        self.cap_ids = []
        with open(capfile, 'r') as reader:
            for line in reader.readlines():
                cap_id, caption = line.strip().split(' ', 1)
                self.captions[cap_id] = caption
                self.cap_ids.append(cap_id)

        self.length = len(self.cap_ids)

    def __getitem__(self, index):
        cap_id = self.cap_ids[index]
        if self.mask:
            caption = self.get_caption_by_id_mask(cap_id)
        else:
            caption = self.get_caption_by_id(cap_id)
        return caption, index, cap_id

    def get_caption_by_id(self, cap_id):
        caption = self.captions[cap_id]
        return caption

    def get_caption_by_id_mask(self, cap_id):
        caption = self.captions[cap_id].split()
        length = len(caption)
        drop_index = random.sample(range(len(caption)), int(0.15*length))
        keep_caption = ' '.join([caption[i] for i in range(length) if i not in drop_index])
        return keep_caption

    def __len__(self):
        return self.length


class TextDatasetWithCapFeat(TextDataset):

    def __init__(self, params):
        super().__init__(params)
        
        self.cap_feature_names = params['cap_feature_names']  # 不同的caption特征名 列表
        self.cap_feature_file_paths = params['cap_feature_file_paths'] # 不同的caption特征文件路径 列表
        self.cap_feature_files = {}
        for i, cap_feature_name in enumerate(self.cap_feature_names):
            self.cap_feature_files[cap_feature_name] = BigFile(self.cap_feature_file_paths[i]) 

    def __getitem__(self, index):
        cap_id = self.cap_ids[index]
        if self.mask:
            caption = self.get_caption_by_id_mask(cap_id)
        else:
            caption = self.get_caption_by_id(cap_id)

        cap_features ={}
        for cap_feature_name in self.cap_feature_files.keys():
            cap_features[cap_feature_name] = self.cap_feature_files[cap_feature_name].read_one(cap_id)

        return caption, index, cap_id, cap_features
    


class PairDataset(data.Dataset):

    def __init__(self, params):
        self.visData = VisionDataset(params)
        self.txtData = TextDataset(params)

        self.cap_ids = self.txtData.cap_ids
        self.length = len(self.cap_ids)

    def __getitem__(self, index):
        cap_id = self.cap_ids[index]
        vis_id = self.get_visId_by_capId(cap_id)

        caption = self.txtData.get_caption_by_id(cap_id)
        vis_feat = self.visData.get_feat_by_id(vis_id)
        return vis_feat, caption, index, vis_id, cap_id

    def get_visId_by_capId(self, cap_id):
        vis_id = cap_id.split('#', 1)[0]
        return vis_id

    def __len__(self):
        return self.length

# class PairDatasetWithCapFeat(PairDataset):

#     def __init__(self, params):
#         super().__init__(params)
#         self.cap_feature_names = params['cap_feature_names']  # 不同的caption特征名 列表
#         self.cap_feature_file_paths = params['cap_feature_file_paths'] # 不同的caption特征文件路径 列表
#         self.cap_feature_files = {}
#         for i, cap_feature_name in enumerate(self.cap_feature_names):
#             self.cap_feature_files[cap_feature_name] = BigFile(self.cap_feature_file_paths[i])
     
#     def __getitem__(self, index):
#         cap_id = self.cap_ids[index]
#         vis_id = self.get_visId_by_capId(cap_id)

#         caption = self.txtData.get_caption_by_id(cap_id)
#         vis_feat = self.visData.get_feat_by_id(vis_id)
        
#         cap_features ={}
#         for cap_feature_name in self.cap_feature_files.keys():
#             cap_features[cap_feature_name] = self.cap_feature_files[cap_feature_name].read_one(cap_id)

#         return vis_feat, caption, index, vis_id, cap_id, cap_features



class FramePairDataset(PairDataset):

    def __init__(self, params):
        self.visData = FrameDataset(params)
        self.txtData = TextDataset(params)

        self.cap_ids = self.txtData.cap_ids
        self.length = len(self.cap_ids)



def vis_provider(params):
    data_loader = torch.utils.data.DataLoader(dataset=VisionDataset(params),
                                              batch_size=params.get('batch_size', 1),
                                              shuffle=params.get('shuffle', False),
                                              pin_memory=params.get('pin_memory', False),
                                              num_workers=params.get('num_workers', 0),
                                              collate_fn=collate_vision)
    return data_loader

def frame_provider(params):
    data_loader = torch.utils.data.DataLoader(dataset=FrameDataset(params),
                                              batch_size=params.get('batch_size', 1),
                                              shuffle=params.get('shuffle', False),
                                              pin_memory=params.get('pin_memory', False),
                                              num_workers=params.get('num_workers', 0),
                                              collate_fn=collate_frames)
    return data_loader


def txt_provider(params):   
    data_loader = torch.utils.data.DataLoader(dataset=TextDataset(params),
                                              batch_size=params.get('batch_size', 1),
                                              shuffle=params.get('shuffle', False),
                                              pin_memory=params.get('pin_memory', False),
                                              num_workers=params.get('num_workers', 0),
                                              collate_fn=collate_text)
    return data_loader

# def txt_provider_with_cap_feat(params):
#     data_loader = torch.utils.data.DataLoader(dataset=TextDatasetWithCapFeat(params),
#                                               batch_size=params.get('batch_size', 1),
#                                               shuffle=params.get('shuffle', False),
#                                               pin_memory=params.get('pin_memory', False),
#                                               num_workers=params.get('num_workers', 0),
#                                               collate_fn=collate_text_with_cap_feat)
#     return data_loader

def pair_provider(params):
    data_loader = torch.utils.data.DataLoader(dataset=PairDataset(params),
                                              batch_size=params.get('batch_size', 1),
                                              shuffle=params.get('shuffle', False),
                                              pin_memory=params.get('pin_memory', False),
                                              num_workers=params.get('num_workers', 0),
                                              collate_fn=collate_pair)
    return data_loader

# def pair_provider_with_cap_feat(params):
#     data_loader = torch.utils.data.DataLoader(dataset=PairDatasetWithCapFeat(params),
#                                               batch_size=params.get('batch_size', 1),
#                                               shuffle=params.get('shuffle', False),
#                                               pin_memory=params.get('pin_memory', False),
#                                               num_workers=params.get('num_workers', 0),
#                                               collate_fn=collate_pair_with_cap_feat)
#     return data_loader


def frame_pair_provider(params):
    data_loader = torch.utils.data.DataLoader(dataset=FramePairDataset(params),
                                              batch_size=params.get('batch_size', 1),
                                              shuffle=params.get('shuffle', False),
                                              pin_memory=params.get('pin_memory', False),
                                              num_workers=params.get('num_workers', 0),
                                              collate_fn=collate_frame_pair)
    return data_loader


if __name__ == '__main__':
    import os
    data_path = os.path.expanduser('/home/zhoufm/github/sea/data')
    # collection = 'tgif-msrvtt10k'
    collection = 'msvd'
    #vid_feat = 'mean_resnext101_resnet152'
    vid_feat = 'mean_resnext101_resnet152'
    vid_feat_dir = os.path.join(data_path, collection, 'FeatureData', vid_feat)

    #vis_loader = vis_provider({'vis_feat': vid_feat_dir, 'batch_size':100, 'num_workers':2})
    #vis_loader = frame_provider({'vis_feat': vid_feat_dir, 'batch_size':1, 'num_workers':0})
    
    #for i, (feat_vecs, idxs, vis_ids) in enumerate(vis_loader):
    #    print i, feat_vecs[0].shape, len(idxs), vis_ids
    #    break
    capfile = os.path.join(data_path, collection, 'TextData', '%s.caption.txt' % collection)
    cap_feature_names = ['bert_feature_Layer_-2_uncased_L-12_H-768_A-12', 'word2vec_flickr_vec500flickr30m_nsw']
    cap_feature_file_paths = [os.path.join(data_path, collection, 'TextData', 'PrecomputedSentFeat', cap_feature_name)  for cap_feature_name in cap_feature_names]


    
    # txt_loader = txt_provider({'capfile':capfile, 'batch_size':100, 'num_workers':2,'caption_mask':True})
    
    # for i, (captions, idxs, cap_ids) in enumerate(txt_loader):
    #     print( i, captions, len(cap_ids))
    #     #print [len(cap) for cap in captions]
    #     break
    
    # exit(0)
    # capfile = os.path.join(data_path, collection, 'TextData', '%s.caption.txt' % collection)
    
    # txt_loader = txt_provider({'capfile':capfile, 'batch_size':100, 'num_workers':2})
    
    # for i, (captions, idxs, cap_ids) in enumerate(txt_loader):
    #     print( i, captions, len(cap_ids))
    #     print( [len(cap) for cap in captions])
    #     break
    bigfile_b = BigFile(cap_feature_file_paths[0])
    bigfile_w = BigFile(cap_feature_file_paths[1])

    pair_loader = pair_provider_with_cap_feat({'vis_feat': vid_feat_dir, 'capfile': capfile, 'cap_feature_names':cap_feature_names, 'cap_feature_file_paths':cap_feature_file_paths, 'batch_size':100, 'num_workers':2, 'shuffle':True})
    for i, (vis_feats, captions, idxs, vis_ids, cap_ids, cap_features) in enumerate(pair_loader):
        print( i, vis_feats.shape, captions[:3], len(cap_ids))
        print( [len(cap) for cap in captions])

        for index, cap_id in enumerate(cap_ids):
            assert bigfile_b.read_one(cap_id) == cap_features[index]['bert_feature_Layer_-2_uncased_L-12_H-768_A-12']
            assert bigfile_w.read_one(cap_id) == cap_features[index]['word2vec_flickr_vec500flickr30m_nsw']

        
        # for i in cap_features:
        #     print(i.keys())
        break
    
    txt_loader = txt_provider_with_cap_feat({'capfile': capfile, 'cap_feature_names':cap_feature_names, 'cap_feature_file_paths':cap_feature_file_paths, 'batch_size':100, 'num_workers':2, 'shuffle':True})
    for i, (txt_input, idxs, cap_ids, cap_features) in enumerate(txt_loader):
        print( i, txt_input[:3], len(cap_ids))
        print( [len(cap) for cap in txt_input])

        for index, cap_id in enumerate(cap_ids):
            assert bigfile_b.read_one(cap_id) == cap_features[index]['bert_feature_Layer_-2_uncased_L-12_H-768_A-12']
            assert bigfile_w.read_one(cap_id) == cap_features[index]['word2vec_flickr_vec500flickr30m_nsw']

        
        # for i in cap_features:
        #     print(i.keys())
        break
    
    
    # pair_loader = pair_provider_with_cap_feat({})
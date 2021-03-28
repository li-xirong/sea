import os
import sys
import time
import json
import pickle
import logging
import argparse

import torch
import numpy as np

import util
import evaluation
import data_provider as data
from common import *
from model import get_model
from bigfile import BigFile
from generic_utils import Progbar


def parse_args():
    parser = argparse.ArgumentParser('SEA predictor')
    parser.add_argument('--rootpath',
                        type=str,
                        default=ROOT_PATH,
                        help='path to datasets. (default: %s)' % ROOT_PATH)
    parser.add_argument('testCollection', type=str, help='test collection')
    parser.add_argument('model_path', type=str, help='Path to load the model.')
    parser.add_argument(
        'sim_name',
        type=str,
        help='sub-folder where computed similarities are saved')
    parser.add_argument('--overwrite',
                        type=int,
                        default=0,
                        choices=[0, 1],
                        help='overwrite existed vocabulary file. (default: 0)')
    parser.add_argument(
        '--query_sets',
        type=str,
        default='tv16.avs.txt',
        help=
        'test query sets,  tv16.avs.txt,tv17.avs.txt,tv18.avs.txt for TRECVID 16/17/18 and tv19.avs.txt for TRECVID19.'
    )
    parser.add_argument('--batch_size',
                        default=128,
                        type=int,
                        help='size of a predicting mini-batch.')
    parser.add_argument('--num_workers',
                        default=2,
                        type=int,
                        help='Number of data loader workers.')
    parser.add_argument('--pre_norm',
                        type=int,
                        default=0,
                        choices=[0, 1],
                        help='whether do l2norm before concat features')
    parser.add_argument(
        '--config_name',
        type=str,
        default='mean_pyresnext-101_rbps13k',
        help='model configuration file. (default: mean_pyresnext-101_rbps13k')
    parser.add_argument(
        '--save_ranking_result',
        type=bool,
        default=False,
        help='whether to save the similarity ranking result after prediction')

    args = parser.parse_args()
    return args


def main():
    opt = parse_args()
    print(json.dumps(vars(opt), indent=2))

    rootpath = opt.rootpath
    testCollection = opt.testCollection

    resume_file = os.path.join(opt.model_path)
    if not os.path.exists(resume_file):
        logging.info(resume_file + ' not exists.')
        sys.exit(0)

    # Load checkpoint
    logger.info('loading model...')
    checkpoint = torch.load(resume_file)
    epoch = checkpoint['epoch']
    best_perf = checkpoint['best_perf']
    config = checkpoint['config']

    if hasattr(config, 't2v_w2v'):
        # if hasattr(config, 'w2v_data_path'):
        #     w2v_data_path = config.w2v_data_path
        # else:
            # w2v_data_path = os.path.join(rootpath, 'word2vec', 'flickr', 'vec500flickr30m')
            # w2v_data_path = os.path.join(rootpath, 'word2vec', 'w2v-flickr-mini')
        w2v_data_path = os.path.join(rootpath, 'word2vec', 'w2v-flickr-mini')
        w2v_feature_file = os.path.join(w2v_data_path, 'feature.bin')
        config.t2v_w2v.w2v.binary_file = w2v_feature_file
    
    cap_feat_names = []
    for encoding in config.text_encoding.split('@'):
        if 'precomputed_bert' in encoding:
            cap_feat_names.append(config.bert_feat_name)

    # if hasattr(config, 'precomputed_feat_bert'):
    #     config.precomputed_feat_bert.binary_file = get_txt2vec('precomputed_bert')(trainCollection, config.bert_feat_name, rootpath)

    # Construct the model
    if not hasattr(config, 'bidirectional'):
        config.bidirectional = False
    if not hasattr(config, 'rnn_dropout'):
        config.rnn_dropout = 0
    config.pre_norm = opt.pre_norm
    if hasattr(config, 'model'):
        model = get_model(config.model)(config)
    else:
        model = get_model('w2vvpp')(config)
    print(model.vis_net)
    print(model.txt_net)
    vis_net_params = sum(p.numel() for p in model.vis_net.parameters())
    txt_net_params = sum(p.numel() for p in model.txt_net.parameters())
    print('    VisNet params: %.2fM' % (vis_net_params / 1000000.0))
    print('    TxtNet params: %.2fM' % (txt_net_params / 1000000.0))
    print('    Total params: %.2fM' %
          ((vis_net_params + txt_net_params) / 1000000.0))
    # print(model.state_dict())
    # print( checkpoint['model'])
    # return
    model.load_state_dict(checkpoint['model'])
    print("=> loaded checkpoint '{}' (epoch {}, best_perf {})".format(
        resume_file, epoch, best_perf))

    #config.vid_feat = 'pyresnext-101_rbps13k,flatten0_output,os+pyresnet-152_imagenet11k,flatten0_output,os'
    vis_feat_file = BigFile(
        os.path.join(rootpath, testCollection, 'FeatureData', config.vid_feat))

    vis_ids = list(
        map(
            str.strip,
            open(
                os.path.join(rootpath, testCollection, 'VideoSets',
                             testCollection + '.txt'))))
    # vis_ids = map(str.strip, open("/data/home/zhoufm/VisualSearch/v3c1/VideoSets/v3c1_of_two_people_kissing.txt"))
    # print ("vis_ids", "/data/home/zhoufm/VisualSearch/v3c1/VideoSets/v3c1_of_two_people_kissing.txt")

    if hasattr(config,
               'model') and config.model in ['multispace_visnetvlad_bow_w2v']:
        vis_loader = data.frame_provider({
            'vis_feat': vis_feat_file,
            'vis_ids': vis_ids,
            'pin_memory': True,
            'batch_size': opt.batch_size,
            'num_workers': opt.num_workers
        })
    else:
        vis_loader = data.vis_provider({
            'vis_feat': vis_feat_file,
            'vis_ids': vis_ids,
            'pin_memory': True,
            'batch_size': opt.batch_size,
            'num_workers': opt.num_workers
        })

    vis_embs = None

    for query_set in opt.query_sets.split(','):
        output_dir = os.path.join(rootpath, testCollection, 'SimilarityIndex',
                                  query_set, opt.sim_name)
        pred_result_file = os.path.join(output_dir, 'id.sent.score.txt')

        if util.checkToSkip(pred_result_file, opt.overwrite):
            continue
        util.makedirs(output_dir)

        #if vis_embs is None:
        #    logger.info('Encoding videos')
        #    vis_embs, vis_ids = evaluation.encode_vis(model, vis_loader)

        capfile = os.path.join(rootpath, testCollection, 'TextData', query_set)
        if testCollection in ['v3c1', 'iacc.3']:
            cap_feat_file_paths = [os.path.join(rootpath, testCollection, 'TextData', 'PrecomputedSentFeat', '%s.%s'%(query_set, cap_feat_name)) for cap_feat_name in cap_feat_names]
        else:
            cap_feat_file_paths = [os.path.join(rootpath, testCollection, 'TextData', 'PrecomputedSentFeat', cap_feat_name) for cap_feat_name in cap_feat_names]
        
        # load text data
        # txt_loader = data.txt_provider({
        #     'capfile': capfile,
        #     'pin_memory': True,
        #     'batch_size': opt.batch_size,
        #     'num_workers': opt.num_workers
        # })
        
        txt_loader = data.txt_provider_with_cap_feat({
            'capfile': capfile,
            'cap_feature_names': cap_feat_names, 
            'cap_feature_file_paths':cap_feat_file_paths,
            'pin_memory': True,
            'batch_size': opt.batch_size,
            'num_workers': opt.num_workers
        })

        opt.save_embs = False
        if opt.save_embs:
            txt_embs, vis_embs = None, None
            for i,(captions, _, _, cap_features) in enumerate(txt_loader):
                txt_emb = model.txt_net(captions, cap_features)
                # import pdb; pdb.set_trace()
                txt_embs = txt_emb if i==0 else torch.cat((txt_embs, txt_emb), dim=0)
                
                
            for j, (vis_input, _, _) in enumerate(vis_loader):
                vis_emb = model.vis_net(vis_input).cpu()
                vis_embs = vis_emb if j==0 else torch.cat((vis_embs, vis_emb), dim=0)
            import pdb; pdb.set_trace()
            txt_embeds_file = os.path.join(rootpath, testCollection, 'Emdebs', 'txt_embs.pth')
            vis_embeds_file = os.path.join(rootpath, testCollection, 'Emdebs', 'vis_embs.pth')
             
            torch.save(txt_embs, txt_embeds_file)  
            torch.save(vis_embs, vis_embeds_file)  
            
            exit(0)

        #logger.info('Encoding %s captions' % query_set)
        #txt_embs, txt_ids = evaluation.encode_txt(model, txt_loader)

        #t2i_matrix = evaluation.compute_sim(txt_embs, vis_embs, measure=config.measure)
        #inds = np.argsort(t2i_matrix, axis=1)

        logger.info('Model prediction ...')
        t2i_matrix, txt_ids, vis_ids = model.predict(txt_loader, vis_loader)
        inds = np.argsort(t2i_matrix, axis=1)

        if testCollection not in ['iacc.3', 'v3c1']:
                # 'msrvtt10ktest', 'tv2016train', 'ht100mmsrvtt10ktest',
                # 'meemsrvtt10ktest', 'msvdtest', 'tgiftest'
        
            label_matrix = np.zeros(inds.shape)
            for index in range(inds.shape[0]):
                ind = inds[index][::-1]
                label_matrix[index][np.where(
                    np.array(vis_ids)[ind] == txt_ids[index].split('#')[0])
                                    [0]] = 1

            (r1, r5, r10, medr, meanr, mir, mAP,
             negRank) = evaluation.eval(label_matrix)
            sum_recall = r1 + r5 + r10
            tempStr = " * Text to video:\n"
            tempStr += " * r_1_5_10: {}\n".format(
                [round(r1, 3), round(r5, 3),
                 round(r10, 3)])
            tempStr += " * medr, meanr, mir: {}\n".format(
                [round(medr, 3),
                 round(meanr, 3),
                 round(mir, 3)])
            tempStr += " * mAP: {}\n".format(round(mAP, 3))
            tempStr += " * " + '-' * 10

            # Video to text
            i2t_matrix = t2i_matrix.T
            inds = np.argsort(i2t_matrix, axis=1)
            label_matrix = np.zeros(inds.shape)
            txt_ids = [txt_id.split('#')[0] for txt_id in txt_ids]
            for index in range(inds.shape[0]):
                ind = inds[index][::-1]
                label_matrix[index][np.where(
                    np.array(txt_ids)[ind] == vis_ids[index])[0]] = 1

            (r1, r5, r10, medr, meanr, mir, mAP,
             negRank) = evaluation.eval(label_matrix)
            sum_recall = r1 + r5 + r10
            tempStr += "\n * Video to text:\n"
            tempStr += " * r_1_5_10: {}\n".format(
                [round(r1, 3), round(r5, 3),
                 round(r10, 3)])
            tempStr += " * medr, meanr, mir: {}\n".format(
                [round(medr, 3),
                 round(meanr, 3),
                 round(mir, 3)])
            tempStr += " * mAP: {}\n".format(round(mAP, 3))
            tempStr += " * " + '-' * 10

            print(tempStr)
            open(os.path.join(output_dir, 'perf.txt'), 'w').write(tempStr)
            util.perf_txt_to_excel('perf_pattern.txt', output_dir) # perf.txt to perf.xlsx
            if not opt.save_ranking_result:
                return

        start = time.time()
        logger.info('Save result ...')
        pbar = Progbar(inds.shape[0])
        with open(pred_result_file, 'w') as fout:
            for index in range(inds.shape[0]):
                ind = inds[index][::-1]

                fout.write(txt_ids[index] + ' ' + ' '.join(
                    [vis_ids[i] + ' %s' % t2i_matrix[index][i]
                     for i in ind]) + '\n')
                pbar.add(1)
        print('writing result into file time: %.3f seconds\n' %
              (time.time() - start))


if __name__ == '__main__':
    main()

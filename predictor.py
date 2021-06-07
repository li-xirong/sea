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
from model import get_model, SentFeatBase
from bigfile import BigFile
from generic_utils import Progbar
from txt2vec import W2Vec


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
                        default=128*8,
                        type=int,
                        help='Size of a predicting mini-batch.')
    parser.add_argument('--num_workers',
                        default=2,
                        type=int,
                        help='Number of data loader workers.')
    parser.add_argument('--pre_norm',
                        type=int,
                        default=0,
                        choices=[0, 1],
                        help='whether do l2norm before concat features')
    # parser.add_argument(
    #     '--config_name',
    #     type=str,
    #     default='mean_pyresnext-101_rbps13k',
    #     help='model configuration file. (default: mean_pyresnext-101_rbps13k')
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
        w2v_data_path = os.path.join(rootpath, 'word2vec', 'w2v-flickr-mini')
        #w2v_feature_file = os.path.join(w2v_data_path, 'feature.bin')
        #config.t2v_w2v.w2v.binary_file = w2v_feature_file
        config.t2v_w2v = W2Vec(w2v_data_path)
    
    for encoding in config.text_encoding.split('@'):
        if encoding == 'bert_precomputed':
            bert_file_dir_list = []
            if testCollection not in ['iacc.3', 'v3c1']:
                bert_file_dir_list.append(os.path.join(rootpath, testCollection, 'SentFeatureData', '%s.caption.txt' % testCollection))
            else:
                for query_set in opt.query_sets.split(','):
                    bert_file_dir_list.append(rootpath, testCollection, 'SentFeatureData', query_set)
            config.bert_feat_base = SentFeatBase(bert_file_dir_list , config.bert_feat_name)
            
    # Construct the model
    if not hasattr(config, 'bidirectional'):
        config.bidirectional = False
    if not hasattr(config, 'rnn_dropout'):
        config.rnn_dropout = 0
    config.pre_norm = opt.pre_norm

    if hasattr(config, 'model'):
        model = get_model(config.model)(config)
    else:
        model = get_model('sea')(config)
    print(model.vis_net)
    print(model.txt_net)
    # calculate the number of parameters
    vis_net_params = sum(p.numel() for p in model.vis_net.parameters())
    txt_net_params = sum(p.numel() for p in model.txt_net.parameters())
    print('    VisNet params: %.2fM' % (vis_net_params / 1000000.0))
    print('    TxtNet params: %.2fM' % (txt_net_params / 1000000.0))
    print('    Total params: %.2fM' %
          ((vis_net_params + txt_net_params) / 1000000.0))

    model.load_state_dict(checkpoint['model'])
    print("=> loaded checkpoint '{}' (epoch {}, best_perf {})".format(
        resume_file, epoch, best_perf))

    vis_feat_file = BigFile(
        os.path.join(rootpath, testCollection, 'FeatureData', config.vid_feat))

    vis_ids = list(
        map(
            str.strip,
            open(
                os.path.join(rootpath, testCollection, 'VideoSets',
                             testCollection + '.txt'))))

    vis_loader = data.vis_provider({
        'vis_feat': vis_feat_file,
        'vis_ids': vis_ids,
        'pin_memory': True,
        'batch_size': opt.batch_size,
        'num_workers': opt.num_workers
    })

    for query_set in opt.query_sets.split(','):
        output_dir = os.path.join(rootpath, testCollection, 'SEA_predict_results',
                                  query_set, opt.sim_name)
        pred_result_file = os.path.join(output_dir, 'id.sent.score.txt')

        if util.checkToSkip(pred_result_file, opt.overwrite):
            continue
        util.makedirs(output_dir)

        capfile = os.path.join(rootpath, testCollection, 'TextData', query_set)
 
        # load text data
        txt_loader = data.txt_provider({
            'capfile': capfile,
            'pin_memory': True,
            'batch_size': opt.batch_size,
            'num_workers': opt.num_workers
        })


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

            (r1, r5, r10, medr, meanr, mir, mAP) = evaluation.eval(label_matrix)
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

            (r1, r5, r10, medr, meanr, mir, mAP) = evaluation.eval(label_matrix)
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
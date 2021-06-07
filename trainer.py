import os
import sys
import time
import json
import shutil
import pickle
import logging
import argparse
import importlib

import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter

import util
import evaluation
import data_provider as data
from common import *
from bigfile import BigFile
from txt2vec import get_txt2vec
from generic_utils import Progbar
from model import get_model, get_we, SentFeatBase


def parse_args():
    parser = argparse.ArgumentParser('SEA training script.')
    parser.add_argument('--rootpath',
                        type=str,
                        default=ROOT_PATH,
                        help='path to datasets. (default: %s)' % ROOT_PATH)
    parser.add_argument('trainCollection', type=str, help='train collection')
    parser.add_argument('valCollection',
                        type=str,
                        help='validation collection')
    parser.add_argument('--overwrite',
                        type=int,
                        default=0,
                        choices=[0, 1],
                        help='overwrite existed vocabulary file. (default: 0)')
    parser.add_argument('--evaluate',
                        '-e',
                        action='store_true',
                        help='evaluate pre-trained model on validation set')
    parser.add_argument('--resume', default='', help='resume')
    parser.add_argument(
        '--val_set',
        type=str,
        default='',
        help='validation collection set (setA, setB). (default: setA)')
    parser.add_argument(
        '--metric',
        type=str,
        default='mir',
        choices=['r1', 'r5', 'medr', 'meanr', 'mir', 'sum_recall'],
        help='performance metric on validation set')
    parser.add_argument('--num_epochs',
                        default=80,
                        type=int,
                        help='Number of training epochs.')
    parser.add_argument('--batch_size',
                        default=128,
                        type=int,
                        help='Size of a training mini-batch.')
    parser.add_argument('--workers',
                        default=2,
                        type=int,
                        help='Number of data loader workers.')
    parser.add_argument('--model_prefix',
                        default='runs_0',
                        type=str,
                        help='Path to save the model and Tensorboard log.')
    parser.add_argument(
        '--config_name',
        type=str,
        default='mean_pyresnext-101_rbps13k',
        help='model configuration file. (default: mean_pyresnext-101_rbps13k')
    parser.add_argument('--pre_norm',
                        type=int,
                        default=0,
                        choices=[0, 1],
                        help='whether do l2norm before concat features')
    parser.add_argument('--save_negative',
                        type=int,
                        default=0,
                        choices=[0, 1],
                        help='whether save negative examples during training')

    args = parser.parse_args()
    return args

def load_config(config_path):
    module = importlib.import_module(config_path)
    return module.config()

def main():
    global opt
    opt = parse_args()
    print(json.dumps(vars(opt), indent=2))

    rootpath = opt.rootpath
    trainCollection = opt.trainCollection
    valCollection = opt.valCollection
    val_set = opt.val_set
    config = load_config('configs.%s' % opt.config_name)

    model_path = os.path.join(rootpath, trainCollection, 'Models',
                              valCollection, val_set, opt.config_name,
                              opt.model_prefix)
    if util.checkToSkip(os.path.join(model_path, 'model_best.pth.tar'),
                        opt.overwrite or opt.evaluate):
        sys.exit(0)
    util.makedirs(model_path)

    global writer
    writer = SummaryWriter(log_dir=model_path, flush_secs=5)

    collections = {'train': trainCollection, 'val': valCollection}
    # print(collections)
   
    capfiles = {'train': '%s.caption.txt' % trainCollection}
    cap_file_paths = {
        'train':
        os.path.join(rootpath, trainCollection, 'TextData', capfiles['train'])
    }

    vis_feat_files = {
        c: BigFile(
            os.path.join(rootpath, collections[c], 'FeatureData',
                         config.vid_feat))
        for c in collections
    }
    config.vis_fc_layers = list(map(int, config.vis_fc_layers.split('-')))
    if config.vis_fc_layers[0] == 0:
        config.vis_fc_layers[0] = vis_feat_files['train'].ndims

    if opt.resume:
        resume_trainCollection = opt.resume.strip('/').split('/')[len(
            rootpath.strip('/').split('/'))]
        rnn_vocab_file = os.path.join(rootpath, resume_trainCollection,
                                      'TextData', 'vocab',
                                      'gru_%d.pkl' % (config.threshold))
    else:
        rnn_vocab_file = os.path.join(rootpath, trainCollection, 'TextData',
                                      'vocab',
                                      'gru_%d.pkl' % (config.threshold))

    # We use a mini set of the whole word2vec model pretrained on flickr data.
    # If you are explore a new dataset, consider use the complete one.
    w2v_data_path = os.path.join(rootpath, 'word2vec', 'w2v-flickr-mini')

    # sent_feat_names: all the precomputed sentence features, such as 'bert_feature_Layer_-2_uncased_L-12_H-768_A-12'
    config.sent_feat_names = []
    # example: config.txt_net_list = ['bow', 'w2v', 'bigru', 'bert']
    config.txt_net_list = [] 

    for encoding in config.text_encoding.split('@'):
        config.txt_net_list.append(encoding.split('_')[0])

    for encoding in config.text_encoding.split('@'):
        if 'bow' in encoding:
            if opt.resume:
                '''
                If your model was pretrained on dataset A and now you want to fine-tune it on dataset B,
                you should use the bag-of-word vocabulary of dataset A.
                Because the vocabulary size of 'bag-of'word' of your model depends on dataset A.
                '''
                resume_trainCollection = opt.resume.strip('/').split('/')[len(rootpath.strip('/').split('/'))]
                bow_vocab_file = os.path.join(rootpath, resume_trainCollection, 'TextData', 'vocab', '%s_%d.pkl'%(encoding, config.threshold))

            else:
                bow_vocab_file = os.path.join(
                    rootpath, trainCollection, 'TextData', 'vocab',
                    '%s_%d.pkl' % (encoding, config.threshold))
            config.t2v_bow = get_txt2vec(encoding)(bow_vocab_file,
                                                   norm=config.bow_norm)
                                                   
        if 'gru' in encoding or 'lstm' in encoding:
            rnn_encoding, config.pooling = encoding.split('_', 1)
            config.t2v_idx = get_txt2vec('idxvec')(rnn_vocab_file)
            if config.we_dim == 500:
                config.we = get_we(config.t2v_idx.vocab, w2v_data_path) # word embedding for RNN
        
        if 'bert_precomputed' in encoding:      
            config.sent_feat_names.append(config.bert_feat_name)
            bert_file_dir_list = []
            bert_file_dir_list.append(os.path.join(rootpath, trainCollection, 'SentFeatureData', '%s.caption.txt' % trainCollection))
            bert_file_dir_list.append(os.path.join(rootpath, valCollection, 'SentFeatureData', val_set, '%s.caption.txt' % valCollection))

            config.bert_feat_base = SentFeatBase(bert_file_dir_list , config.bert_feat_name)
            
        if 'w2v' in encoding or 'netvlad' in encoding:
            config.t2v_w2v = get_txt2vec(encoding)(w2v_data_path)
            config.w2v_out_size = config.t2v_w2v.ndims

    config.txt_fc_layers = list(map(int, config.txt_fc_layers.split('-')))

    # Construct and the model and print the details
    config.pre_norm = opt.pre_norm
    if not hasattr(config, 'bidirectional'):
        config.bidirectional = False
    if not hasattr(config, 'rnn_dropout'):
        config.rnn_dropout = 0

    if hasattr(config, 'model'):
        model = get_model(config.model)(config)
    else:
        model = get_model('sea')(config)
    print(model.vis_net)
    print(model.txt_net)
    vis_net_params = sum(p.numel() for p in model.vis_net.parameters())
    txt_net_params = sum(p.numel() for p in model.txt_net.parameters())
    print('    VisNet params: %.2fM' % (vis_net_params / 1000000.0))
    print('    TxtNet params: %.2fM' % (txt_net_params / 1000000.0))
    print('    Total params: %.2fM' %
          ((vis_net_params + txt_net_params) / 1000000.0))

    caption_mask = ('caption_mask' in opt.config_name)
    logger.info('caption mask: %s' % caption_mask)

    val_vis_ids = list(
        map(
            str.strip,
            open(
                os.path.join(rootpath, collections['val'], 'VideoSets',
                             '%s.txt' % collections['val']))))

    train_loader = data.pair_provider({
        'vis_feat': vis_feat_files['train'],
        'capfile': cap_file_paths['train'],
        'pin_memory': True,
        'batch_size': opt.batch_size,
        'num_workers': opt.workers,
        'shuffle': True,
        'caption_mask': caption_mask
    })
    
    val_vis_loader = data.vis_provider({
        'vis_feat': vis_feat_files['val'],
        'vis_ids': val_vis_ids,
        'pin_memory': True,
        'batch_size': opt.batch_size,
        'num_workers': opt.workers
    })

    if valCollection == 'iacc.3' or valCollection == 'gcc11val':
        cap_file_paths['val'] = os.path.join(rootpath, valCollection,
                                             'TextData', val_set)
        val_txt_loader = data.txt_provider({
            'capfile': cap_file_paths['val'],
            'pin_memory': True,
            'batch_size': opt.batch_size,
            'num_workers': opt.workers
        })
    else:
        cap_file_paths['val'] = os.path.join(rootpath, valCollection,
                                             'TextData', val_set,
                                             '%s.caption.txt' % valCollection)
        
        val_txt_loader = data.txt_provider({
            'capfile': cap_file_paths['val'],
            'pin_memory': True,
            'batch_size': opt.batch_size,
            'num_workers': opt.workers
        })
    
    if opt.evaluate:
        resume_file = os.path.join(model_path, 'model_best.pth.tar')
        assert os.path.exists(resume_file), '%s not exists' % resume_file
        checkpoint = torch.load(resume_file)
        model.load_state_dict(checkpoint['model'])
        print("=> loaded checkpoint '{}' (epoch {}, best_perf {})".format(
            resume_file, checkpoint['epoch'], checkpoint['best_perf']))
        if valCollection == 'iacc.3':
            cur_perf = validate_v3(model, val_txt_loader, val_vis_loader,
                                   checkpoint['epoch'], model_path)
        else:
            cur_perf = validate_v2(model,
                                   val_txt_loader,
                                   val_vis_loader,
                                   checkpoint['epoch'],
                                   metric=opt.metric)
        exit(0)

    best_perf = 0
    if opt.resume:
        assert os.path.exists(opt.resume)
        checkpoint = torch.load(opt.resume)
        model.load_state_dict(checkpoint['model'])
        print("=> loaded checkpoint '{}' (epoch {}, best_perf {})".format(
            opt.resume, checkpoint['epoch'], checkpoint['best_perf']))
        best_perf = checkpoint['best_perf']
        save_checkpoint(
            {
                'epoch': 0,
                'model': model.state_dict(),
                'best_perf': best_perf,
                'config': config,
                'opt': opt
            },
            is_best=True,
            logdir=model_path,
            only_best=True,
            filename='checkpoint_epoch_0.pth.tar')

    # Train the Model
    no_improve_counter = 0
    val_perf_hist_fout = open(os.path.join(model_path, 'val_perf_hist.txt'),
                              'w')
    for epoch in range(opt.num_epochs):

        print('Epoch[{0} / {1}] LR: {2}'.format(epoch, opt.num_epochs,
                                                model.learning_rate))
        print('-' * 10)
        writer.add_scalar('train/learning_rate', model.learning_rate[0], epoch)

        if opt.save_negative:
            with open(os.path.join(model_path, 'max_negative_%d.txt' % epoch),
                      'w') as fw:
                train(model, train_loader, fw)
        else:
            train(model, train_loader, None)

        # evaluate on validation set
        #cur_perf = validate(model, data_loaders['val'], epoch, metric=opt.metric)
        if valCollection == 'iacc.3':
            cur_perf = validate_v3(model, val_txt_loader, val_vis_loader,
                                   epoch, model_path)
        else:
            cur_perf = validate_v2(model,
                                   val_txt_loader,
                                   val_vis_loader,
                                   epoch,
                                   metric=opt.metric)
        model.lr_step(val_value=cur_perf)

        print(' * Current perf: {}\n * Best perf: {}\n'.format(
            cur_perf, best_perf))
        val_perf_hist_fout.write('epoch_%d:\nText2Video(%s): %f\n' %
                                 (epoch, opt.metric, cur_perf))
        val_perf_hist_fout.flush()

        # remember best performance and save checkpoint
        is_best = cur_perf > best_perf
        best_perf = max(cur_perf, best_perf)
        save_checkpoint(
            {
                'epoch': epoch + 1,
                'model': model.state_dict(),
                'best_perf': best_perf,
                'config': config,
                'opt': opt
            },
            is_best,
            logdir=model_path,
            only_best=True,
            filename='checkpoint_epoch_%s.pth.tar' % epoch)
        if is_best:
            no_improve_counter = 0
        else:
            no_improve_counter += 1
            if no_improve_counter > 10:
                print('Early stopping happended.\n')
                break

    val_perf_hist_fout.close()
    message = 'best performance on validation:\n Text to video({}): {}'.format(
        opt.metric, best_perf)
    print(message)
    with open(os.path.join(model_path, 'val_perf.txt'), 'w') as fout:
        fout.write(message)


def train(model, train_loader, fw=None):
    # average meters to record the training statistics
    batch_time = util.AverageMeter()
    data_time = util.AverageMeter()

    # switch to train mode
    model.switch_to_train()

    progbar = Progbar(len(train_loader.dataset))
    end = time.time()
    for _, train_data in enumerate(train_loader):

        data_time.update(time.time() - end)

        vis_feats, captions, _, vis_ids, cap_ids = train_data
        # loss, indices_im = model.train(txt_input, cap_ids, vis_input, vis_ids)
        loss, indices_im = model.train(captions, cap_ids,  vis_feats, vis_ids)

        progbar.add(len(vis_feats),
                    values=[('data_time', data_time.val),
                            ('batch_time', batch_time.val), ('loss', loss)])

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # Record logs in tensorboard
        writer.add_scalar('train/Loss', loss, model.iters)
        
        #if fw is not None:
        #    for j, cap_id in enumerate(cap_ids):
        #        fw.write('%s %s\n' % (cap_id, ' '.join([vis_ids[k] for k in indices_im[j]])))


# def validate(model, val_loader, epoch, metric='mir'):
#     # compute the encoding for all the validation videos and captions
#     ## video retrieval
#     ## Multi-Space
#     #txt2vis_sim = model.validate_similarity(val_loader)

#     #(r1, r5, r10, medr, meanr, mir) = evaluation.eval_qry2retro(txt2vis_sim, n_qry=1)
#     vis_embs, txt_embs, vis_ids, txt_ids = evaluation.encode_data(
#         model, val_loader)

#     keep_vis_order = []
#     keep_vis_ids = []
#     for i, vid in enumerate(vis_ids):
#         if vid not in keep_vis_ids:
#             keep_vis_order.append(i)
#             keep_vis_ids.append(vid)
#     vis_embs = vis_embs[keep_vis_order]
#     vis_ids = keep_vis_ids

#     # video retrieval
#     txt2vis_sim = evaluation.compute_sim(txt_embs, vis_embs)
#     #(r1, r5, r10, medr, meanr, mir) = evaluation.eval_qry2retro(txt2vis_sim, n_qry=1)
#     inds = np.argsort(txt2vis_sim, axis=1)
#     label_matrix = np.zeros(inds.shape)
#     for index in range(inds.shape[0]):
#         ind = inds[index][::-1]
#         label_matrix[index][np.where(
#             np.array(vis_ids)[ind] == txt_ids[index].split('#')[0])[0]] = 1

#     (r1, r5, r10, medr, meanr, mir, mAP) = evaluation.eval(label_matrix)
#     sum_recall = r1 + r5 + r10
#     print(" * Text to video:")
#     print(" * r_1_5_10: {}".format([round(r1, 3),
#                                     round(r5, 3),
#                                     round(r10, 3)]))
#     print(" * medr, meanr, mir: {}".format(
#         [round(medr, 3), round(meanr, 3),
#          round(mir, 3)]))
#     print(" * mAP: {}".format(round(mAP, 3)))
#     print(" * " + '-' * 10)

#     writer.add_scalar('val/r1', r1, epoch)
#     writer.add_scalar('val/r5', r5, epoch)
#     writer.add_scalar('val/r10', r10, epoch)
#     writer.add_scalar('val/medr', medr, epoch)
#     writer.add_scalar('val/meanr', meanr, epoch)
#     writer.add_scalar('val/mir', mir, epoch)
#     writer.add_scalar('val/mAP', mAP, epoch)

#     return locals().get(metric, mir)


def validate_v2(model, txt_loader, vis_loader, epoch, metric='mir'):
    # compute the encoding for all the validation videos and captions
    ## video retrieval
    ## Multi-Space

    txt2vis_sim, txt_ids, vis_ids = model.predict(txt_loader, vis_loader)

    assert txt2vis_sim.shape == (len(txt_ids), len(vis_ids)), 'txt2vis_sim.shape%s not match (txt_ids(%s),vis_ids(%s))' % \
            (txt2vis_sim.shape, len(txt_ids), len(vis_ids))

    inds = np.argsort(txt2vis_sim, axis=1)  

    label_matrix = np.zeros(inds.shape)
    for index in range(inds.shape[0]):  
        ind = inds[index][::-1]  
        label_matrix[index][np.where(
            np.array(vis_ids)[ind] == txt_ids[index].split('#')[0])[0]] = 1

    (r1, r5, r10, medr, meanr, mir, mAP) = evaluation.eval(label_matrix)
    sum_recall = r1 + r5 + r10
    print(" * Text to video:")
    print(" * r_1_5_10: {}".format([round(r1, 3),
                                    round(r5, 3),
                                    round(r10, 3)]))
    print(" * medr, meanr, mir: {}".format(
        [round(medr, 3), round(meanr, 3),
         round(mir, 3)]))
    print(" * mAP: {}".format(round(mAP, 3)))
    print(" * " + '-' * 10)

    vis2txt_sim = txt2vis_sim.T
    inds = np.argsort(vis2txt_sim, axis=1)
    label_matrix = np.zeros(inds.shape)
    txt_ids = [txt_id.split('#', 1)[0] for txt_id in txt_ids]
    for index in range(inds.shape[0]):
        ind = inds[index][::-1]
        label_matrix[index][np.where(
            np.array(txt_ids)[ind] == vis_ids[index])[0]] = 1

    (r1i, r5i, r10i, medri, meanri, miri, mAPi) = evaluation.eval(label_matrix)
    sum_recall += r1i + r5i + r10i
    print(" * Video to text:")
    print(" * r_1_5_10: {}".format(
        [round(r1i, 3), round(r5i, 3),
         round(r10i, 3)]))
    print(" * medr, meanr, mir: {}".format(
        [round(medri, 3), round(meanri, 3),
         round(miri, 3)]))
    print(" * mAP: {}".format(round(mAPi, 3)))
    print(" * " + '-' * 10)

    writer.add_scalar('val/r1', r1, epoch)
    writer.add_scalar('val/r5', r5, epoch)
    writer.add_scalar('val/r10', r10, epoch)
    writer.add_scalar('val/medr', medr, epoch)
    writer.add_scalar('val/meanr', meanr, epoch)
    writer.add_scalar('val/mir', mir, epoch)
    writer.add_scalar('val/mAP', mAP, epoch)

    return locals().get(metric, mir)


def validate_v3(model,
                txt_loader,
                vis_loader,
                epoch,
                model_path,
                metric='infAP'):
    topic_sets = ['tv16', 'tv17', 'tv18']
    pred_treceval_file = os.path.join(model_path, 'id.score.treceval')
    TOPIC_SIZE = 30
    MAX_SCORE = 9999
    MAX_RETURN = 1000
    TEAM = 'RUCMM'
    infAPs = [0] * len(topic_sets)

    t2i_matrix, txt_ids, vis_ids = model.predict(txt_loader, vis_loader)
    inds = np.argsort(t2i_matrix, axis=1)

    for k, topic_set in enumerate(topic_sets):
        topic_ids = [
            line.strip().split()[0] for line in open(
                os.path.join(opt.rootpath, opt.valCollection, 'TextData',
                             '%s.avs.txt' % topic_sets[k]))
        ]
        with open(pred_treceval_file, 'w') as fout:
            for index in range(inds.shape[0])[k * TOPIC_SIZE:(k + 1) *
                                              TOPIC_SIZE]:
                assert (
                    txt_ids[index]
                    in topic_ids), '%s not in %s' % (txt_ids[index], topic_set)
                ind = inds[index][::-1][:MAX_RETURN]

                newlines = [
                    '%s 0 %s %d %d %s' % ('1' + txt_ids[index], vis_ids[idx],
                                          i + 1, MAX_SCORE - i, TEAM)
                    for i, idx in enumerate(ind)
                ]
                fout.write('\n'.join(newlines) + '\n')

        gt_file = os.path.join(opt.rootpath, opt.valCollection, 'TextData',
                               'avs.qrels.%s' % topic_sets[k])
        cmd = 'perl tv-avs-eval/sample_eval.pl -q %s %s' % (gt_file,
                                                            pred_treceval_file)
        res = os.popen(cmd).read().split('\n')
        infAPs[k] = [
            float(line.split()[-1]) for line in res
            if 'infAP' in line and 'all' in line
        ][0]

    print(" * Text to video:")
    print(" * infAP({}): {}".format('_'.join(topic_sets), infAPs))
    print(" * sum_of_infAP: {}".format(sum(infAPs)))
    print(" * " + '-' * 10)

    for i in range(len(topic_sets)):
        writer.add_scalar('val/%s' % topic_sets[i], infAPs[i], epoch)
    writer.add_scalar('val/total', sum(infAPs), epoch)

    return sum(infAPs)


def save_checkpoint(state,
                    is_best,
                    filename='checkpoint.pth.tar',
                    only_best=False,
                    logdir=''):
    resfile = os.path.join(logdir, filename)
    torch.save(state, resfile)
    if is_best:
        shutil.copyfile(resfile, os.path.join(logdir, 'model_best.pth.tar'))

    if only_best:
        os.remove(resfile)


if __name__ == '__main__':
    main()

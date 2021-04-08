'''Generate bert feature.
'''
import os
from bert_serving.client import BertClient
import argparse
bert_serving_ip = '10.77.50.197'


def parse_args():
    parser = argparse.ArgumentParser(
        'sentence to bert feature via "bert-as-service"')

    parser.add_argument('--dataset',
                        type=str,
                        default='msrvtt10k',
                        help='target dataset')
    parser.add_argument('--query_sets', type=str, default='tv16.avs.txt')
    parser.add_argument(
        '--feature_name',
        type=str,
        default='bert_feature_Layer_-2_uncased_L-12_H-768_A-12.txt')
    parser.add_argument('--rootpath',type=str,default='./data')

    args = parser.parse_args()
    return args


def encode_bert_feature(txt_input):
    bc = BertClient(ip=bert_serving_ip,
                    port=1234,
                    port_out=4321,
                    check_version=False)
    txt_bert_feature = bc.encode(txt_input)
    return txt_bert_feature


def main(opt):
    # DATASET = 'msvdtest'
    DATASET = opt.dataset

    print("target dataset is %s." % DATASET)
    if DATASET == 'iacc.3' or DATASET == 'v3c1':
        TXT_FILE = os.path.join('./data/', DATASET, 'TextData', opt.query_sets)
        bert_feature_file = os.path.join(
            './data/', DATASET, 'TextData', opt.query_sets +
            '.bert_feature_Layer_-2_uncased_L-12_H-768_A-12.txt')
    else:
        TXT_FILE = os.path.join('./data/', DATASET, 'TextData',
                                DATASET + '.caption.txt')
        bert_feature_file = os.path.join(
            './data/', DATASET, 'TextData',
            DATASET + '.bert_feature_Layer_-2_uncased_L-12_H-768_A-12.txt')
    print('caption file is %s.' % TXT_FILE)
    print('feature file is %s.' % bert_feature_file)

    print('>>>reading captions>>>')
    txt_input = []
    cap_ids = []
    with open(TXT_FILE, 'r') as fr:
        for line in fr.readlines():
            cap_id, cap_content = line.strip().split(' ', 1)
            cap_ids.append(cap_id)
            txt_input.append(cap_content)
            # break
    print(len(cap_ids), len(txt_input))

    print('>>>encoding sentences>>>')
    txt_bert_feature = encode_bert_feature(txt_input)
    assert len(txt_bert_feature) == len(cap_ids)
    print('enconding finished')

    print('>>>writing to txt file>>>')
    with open(bert_feature_file, 'w') as fw:
        for i in range(len(cap_ids)):
            line = cap_ids[i] + ' ' + ' '.join(
                [str(num) for num in txt_bert_feature[i]]) + '\n'
            fw.write(line)
            # break
    print(">>>txt to bin format>>>")
    os.system(' '.join([
        'python', 'txt2bin.py', '0', bert_feature_file, '0',
        bert_feature_file[:-4]
    ]))


if __name__ == '__main__':
    opt = parse_args()
    print(opt)
    main(opt)
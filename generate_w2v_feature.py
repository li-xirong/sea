from txt2vec import W2VecNSW
import os

rootpath = 'data'
w2v_data_path = os.path.join(rootpath, 'word2vec', 'flickr', 'vec500flickr30m')
t2v_w2v = W2VecNSW(w2v_data_path)

for dataset in ['msvdtest', 'msvdval', 'msvdtrain']:
    txt_file = '/home/zhoufm/github/sea/data/' + dataset + '/TextData/' + dataset + '.caption.txt'
    w2v_feature_file = os.path.join(rootpath, dataset, 'TextData', dataset +'.word2vec_flickr_vec500flickr30m.txt')
    
    print txt_file,
    print w2v_feature_file

    txt_input = []
    cap_ids = []
    with open(txt_file, 'r') as fr:
        for line in fr.readlines():
            cap_id, cap_content = line.strip().split(' ',1)
            cap_ids.append(cap_id)
            txt_input.append(cap_content)
            # break
    print len(cap_ids), len(txt_input)

    txt_w2v_feature = []
    for sentence in txt_input:
        txt_w2v_feature.append(t2v_w2v.encoding(sentence))
    # txt_w2v_feature = t2v_w2v.encoding(txt_input)
    # print txt_bert_feature, txt_bert_feature[0][:10]
    assert len(txt_w2v_feature) == len(cap_ids)

    with open(w2v_feature_file,'w') as fw:
        for i in range(len(cap_ids)):
            line = cap_ids[i] + ' ' + ' '.join([str(num) for num in txt_w2v_feature[i]]) + '\n'
            fw.write(line)
            # break

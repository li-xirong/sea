from bert_serving.client import BertClient
bert_serving_ip = '10.77.50.197'


dataset = 'msvdtest'
txt_file = '/home/zhoufm/github/sea/data/' + dataset + '/TextData/' + dataset + '.caption.txt'
bert_feature_file = '/home/zhoufm/github/sea/data/' + dataset + '/TextData/' + dataset + '.bert_feature_Layer_-2_uncased_L-12_H-768_A-12.txt'
print txt_file

txt_input = []
cap_ids = []
with open(txt_file, 'r') as fr:
    for line in fr.readlines():
        cap_id, cap_content = line.strip().split(' ',1)
        cap_ids.append(cap_id)
        txt_input.append(cap_content)
        # break
print len(cap_ids), len(txt_input)
bc = BertClient(ip=bert_serving_ip, port=1234, port_out=4321, check_version=False)
txt_bert_feature = bc.encode(txt_input)
# print txt_bert_feature, txt_bert_feature[0][:10]
assert len(txt_bert_feature) == len(cap_ids)

with open(bert_feature_file,'w') as fw:
    for i in range(len(cap_ids)):
        line = cap_ids[i] + ' ' + ' '.join([str(num) for num in txt_bert_feature[i]]) + '\n'
        fw.write(line)
        # break

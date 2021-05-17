class config(object):
    model = 'sea'
    
    text_encoding = 'bow_nsw@w2v_nsw@bert_precomputed'

    threshold = 5
    bow_norm = 0
    we_dim = 500
    rnn_size = 1024
    rnn_layer = 1

    bert_out_size = 768
    bert_feat_name = 'bert_feature_Layer_-2_uncased_L-12_H-768_A-12'
    # for oneline bert encoder using 'bert-as-service'
    bert_service_ip = '10.77.50.197'
    bert_port = 1234
    bert_port_out = 4321

    txt_fc_layers = '0-2048'
    txt_norm = 2 # L_2 norm
    use_abs = False
    batch_norm = False

    vid_feat = 'mean_resnext101_resnet152'
    vis_fc_layers = '0-2048'
    vis_norm = 2 # L_2 norm

    # dropout
    dropout = 0.2
    last_dropout = 0.2

    # activation
    activation = 'tanh'
    last_activation = 'tanh'

    # loss
    loss = 'mrl'
    margin = 0.2
    direction = 't2i'       # only valid for mrl loss
    # Use max instead of sum in the rank loss
    max_violation = True    # only valid for mrl loss
    # cost style (sum|mean)
    cost_style = 'sum'      # only valid for mrl loss
    # Similarity measure used (cosine|order)
    measure = 'cosine'

    # optimizer
    optimizer ='rmsprop'
    # Initial learning rate.
    lr = 0.0001
    lr_decay_rate = 0.99
    # Gradient clipping threshold
    grad_clip = 2


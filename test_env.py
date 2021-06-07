from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import random
import numpy as np
import unittest

from common import ROOT_PATH
rootpath = ROOT_PATH

collection = 'msvd'
train_collection = '%strain' % collection
val_collection = '%sval' % collection
test_collection = '%stest' % collection

vis_feat = 'mean_resnext101_resnet152'
bert_txt_feat = 'bert_feature_Layer_-2_uncased_L-12_H-768_A-12'
#vis_feat = 'mean_pyresnext-101_rbps13k,flatten0_output,os'


def get_txt_feat_dir(collection, txt_feat, rootpath):
    return os.path.join(rootpath, collection, 'SentFeatureData', '%s.caption.txt' % collection, txt_feat)


class TestSuite (unittest.TestCase):

    def test_rootpath(self):
        self.assertTrue(os.path.exists(rootpath))

    def test_w2v_dir(self):
        w2v_dir = os.path.join(rootpath, 'word2vec', 'w2v-flickr-mini')
        self.assertTrue(os.path.exists(w2v_dir), 'missing %s'%w2v_dir)

    def test_train_data(self):
        cap_file = os.path.join(rootpath, train_collection, 'TextData', '%s.caption.txt' % train_collection)
        self.assertTrue(os.path.exists(cap_file), 'missing %s'%cap_file)
        feat_dir = os.path.join(rootpath, train_collection, 'FeatureData', vis_feat)
        self.assertTrue(os.path.exists(feat_dir), 'missing %s'%feat_dir)
        
        txt_feat_dir = get_txt_feat_dir(train_collection, bert_txt_feat, rootpath)
        self.assertTrue(os.path.exists(txt_feat_dir), 'missing %s'% txt_feat_dir)

    def test_val_data(self):
        cap_file = os.path.join(rootpath, val_collection, 'TextData', '%s.caption.txt' % val_collection)
        self.assertTrue(os.path.exists(cap_file), 'missing %s'%cap_file)
        feat_dir = os.path.join(rootpath, val_collection, 'FeatureData', vis_feat)
        self.assertTrue(os.path.exists(feat_dir), 'missing %s'%feat_dir)
  
        txt_feat_dir = get_txt_feat_dir(val_collection, bert_txt_feat, rootpath)
        self.assertTrue(os.path.exists(txt_feat_dir), 'missing %s'% txt_feat_dir)

    def test_test_data(self):
        cap_file = os.path.join(rootpath, test_collection, 'TextData', '%s.caption.txt' % test_collection)
        self.assertTrue(os.path.exists(cap_file), 'missing %s'%cap_file)
        feat_dir = os.path.join(rootpath, test_collection, 'FeatureData', vis_feat)
        self.assertTrue(os.path.exists(feat_dir), 'missing %s'%feat_dir)
 
        txt_feat_dir = get_txt_feat_dir(test_collection, bert_txt_feat, rootpath)
        self.assertTrue(os.path.exists(txt_feat_dir), 'missing %s'% txt_feat_dir)


suite = unittest.TestLoader().loadTestsFromTestCase(TestSuite)
unittest.TextTestRunner(verbosity=2).run(suite)
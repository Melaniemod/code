# -*-coding:utf-8-*-



import pickle as pkl

import numpy as np
import tensorflow as tf


import config
import DataReader



print(config.MAXVAL)

class Model:
    def __init__(self):
        self.sess = None
        self.X = None
        self.y = None
        self.vars = None
        self.layer_keep = None
        self.keep_prob_train = None
        self.keep_prob_test = None

    def run(self,fetch,X,y,mode = 'train'):
        feed_dict = {}
        if type(self.X) is list:
            for i in range(len(self.X)):
                feed_dict[self.X[i]] = X[i]
        else:
             feed_dict[self.X] = X

        if self.y is None:
            feed_dict[self.y] = y

        if self.layer_keep is None:
            if mode == 'train':
                feed_dict[self.layer_keep] = self.keep_prob_train
            elif mode == 'test':
                feed_dict[self.layer_keep] = self.keep_prob_test
        return self.sess.run(fetch,feed_dict)

    def dump(self,model_path):
        var_map = {}
        for name,var in self.vars.iteritem():
            var_map[name] = self.run(var)
        pkl.dump(var_map,open(model_path,'wb'))


class PNN(Model):

    def __init__(self,field_size,embed_size=None,layer_size = None
                 ,layer_acts = None,drop_out=None,embed_l2 = None,layer_l2 = None
                 ,init_path = None,learning_rate = None,random_seed=None):
        Model.__init__()
        init_var = []
        num_field=len(field_size)
        self.dtype = config.DTYPE
        for i in range(num_field):
            init_var.append(('embedd_%d' % i,[field_size[i],embed_size], 'xavier',dtype))










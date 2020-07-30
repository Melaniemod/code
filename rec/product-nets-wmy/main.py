# -*-coding:utf-8-*-


import pickle as pkl

import numpy as np
import tensorflow as tf

from python import utils


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




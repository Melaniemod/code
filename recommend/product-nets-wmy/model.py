# -*-coding:utf-8-*-



import pickle as pkl

import numpy as np
import tensorflow as tf


import config
import data_reader import DataReader,activate,get_opt,init_var_map,slice


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
        self.dtype = config.DTYPE
        self.field_size = field_size
        self.embed_size = embed_size
        self.layer_size = layer_size
        self.layer_acts = layer_acts
        self.drop_out = drop_out
        self.embed_l2 = embed_l2
        self.layer_l2 = layer_l2
        self.init_path = init_path
        self.learning_rate = learning_rate
        self.random_seed = random_seed

        self.num_field=len(field_size)
        self.num_pair = int(self.num_field *(self.num_field-1) /2)

        self.init_graph()


    def init_weights(self):
        init_var = []

        # 各种权重初始化
        for i in range(self.num_field):
            init_var.append(('embed_%d' % i,[self.field_size[i],self.embed_size], 'xavier',self.dtype))

        node_in = self.num_field * self.embed_size + self.num_pair
        for i in range(len(self.layer_size)):
            init_var.append(('w%d' % i,[node_in,self.layer_size[i]],'xavier',self.dtype))
            init_var.append(('b%d' % i,[self.layer_size[i]],'xavier',self.dtype))
            node_in = self.layer_size[i]

        return init_var

    def init_graph(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            if self.random_seed is not None:
                tf.set_random_seed(self.random_seed)
            # 定义容器
            self.X = [tf.sparse_placeholder(self.dtype) for _ in range(self.num_field)]
            self.y = tf.placeholder(self.dtype)
            self.keep_drop_out = tf.placeholder(self.dtype)
            self.keep_prob_train = 1-np.array(self.drop_out)
            self.keep_prob_test = np.ones_like(self.drop_out)

            # 初始化权重
            self.vars = init_var_map(self.init_weights())


            embed_w = [self.vars['embed_%d' % i] for i in range(self.num_field)]
            embed = tf.concat([tf.sparse_tensor_dense_matmul(self.X[i],embed_w[i]) for i in range(self.num_field)],1)

            embed = tf.reshape(embed,[-1,self.num_field,self.embed_size])

            row = []
            col = []
            for i in range(self.num_pair-1):
                for j in range(i+1,self.num_pair):
                    row.append(i)
                    col.append(j)
            p = tf.transpose(
                tf.gather(
                    tf.transpose(embed,[1,0,2]),row
                ),[1,0,2]
            )

            q = tf.transpose(
                tf.gather(
                    tf.transpose(embed,[1,0,2]),col
                ),[1,0,2]
            )

            p = tf.reshape(p,[-1,self.num_pair,self.embed_size])
            q = tf.reshape(q,[-1,self.num_pair,self.embed_size])
            pq = tf.reshape(
                tf.reduce_sum(p*q,axis=[-1]),[-1,self.num_pair]
            )

            l = tf.concat([embed,pq],1)

            for i in range(len(self.layer_size)):
                wi = self.vars['w_%d' % i]
                bi = self.vars['b_%d' % i]
                l = tf.nn.dropout(
                    activate(
                        tf.matmul(l,wi) +bi
                        ,self.layer_acts[i]
                    ),self.layer_keep[i]
                )

            l = tf.squeeze(l)

            self.prob = tf.sigmoid(l)
            self.loss = tf.reduce_sum(
                tf.nn.sigmoid_cross_entropy_with_logits(labels=self.y, logits=l)
            )
            if self.layer_l2 is not None:
                self.loss += self.embed_l2 * tf.nn.l2_loss(embed)
                for i in range(len(self.layer_size)):
                    wi = self.vars["w_%d" % i]
                    self.loss += self.layer_l2[i] * tf.nn.l2_loss(wi)
                self.opt = get_opt(self.opt,self.learning_rate,self.loss)

                conf = tf.ConfigProto()
                conf.gpu_options.allow_grath = True
                self.sess = tf.Session(config = conf)
                tf.global_variables_initializer().run(session=self.sess)





























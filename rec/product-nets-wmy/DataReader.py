# -*-coding:utf-8-*-


import sys
if sys.version[0] == '2':
    import cPickle as pkl
else:
    import pickle as pkl

import numpy as np
import tensorflow as tf
from scipy.sparse import coo_matrix

import config


class DataReader():
    """todo 读取数据，并且将长向量转为list，list 中的元素是每个 field 的稀疏向量"""
    def __init__(self,feat_index_file='../data/featindex.txt',
                 train_file=None,test_file = None,
                 field_size = None,
                 output_dim =1
                 ):
        self.feat_index_file = feat_index_file
        self.train_file = train_file
        self.test_file = test_file


        with open(self.feat_index_file) as file:
            for line in file:
                line = line.strip().split(":")
                if len(line)>1:
                    f = int(line[0]) - 1
                    field_size[f] += 1
        self.field_size = field_size
        self.field_offset = [sum(self.field_size[:i]) for i in range(field_size)]
        self.input_dim = sum(self.field_size)
        self.output_dim = output_dim

    def read_data(self):
        y=[]
        X=[]
        D=[]
        with open(self.train_file) as file:
            for line in file:
                line = line.strip().split()
                y_i = int(line[0])
                X_i = [ int(x.split(":")[0]) for x in line[1:]]
                D_i = [ int(x.split(':')[1]) for x in line[1:]]
                y.append(y_i)
                X.append(X_i)
                D.append(D_i)
        y = np.reshape(y,[-1])
        X = self.data_2_coo(zip(X,D),[len(X),self.input_dim]).tocsr()
        return X,y


    def data_2_coo(self,data,shape):
        n=0
        coo_row = []
        coo_col = []
        coo_data = []

        for x,d in data:
            row = [n] * len(x)
            col = x
            data = d
            coo_row.extend(row)
            coo_col.extend(col)
            coo_data.extend(data)

        coo_row = np.array(coo_row)
        coo_col = np.array(coo_col)
        coo_data = np.array(coo_data)
        return coo_matrix((coo_data,(coo_row,coo_col)),shape=shape)


    def shuffer(self,data):
        X,y = data
        ind = np.arange(len(X))
        np.random.shuffle(ind)
        return X[ind],y[ind]


    def split_data(self,data,skip_empty = True):
        fields = []
        for i in range(len(self.field_offset)-1):
            start = self.field_offset[i]
            end = self.field_offset[i+1]
            if start == end and skip_empty:
                continue
            field_i = data[0][:,start:end]
            fields.append(field_i)
        fields.append(data[0][:,self.field_offset[-1]:])
        return fields,data[1]


def csr_2_input(csr_mat):
    if not isinstance(csr_mat,list):
        coo_mat = csr_mat.tocoo()
        index = np.vstack((coo_mat.row,coo_mat.col)).transpose()
        value = coo_mat.data
        shape = coo_mat.shape
        return index,value,shape
    else:
        ls_input = []
        for field in csr_mat:
            ls_input.append(csr_2_input(field))
            return ls_input


def slice(csr_data,start=0,size = -1):
    if not isinstance(csr_data,list):
        if size == -1 or start+size >= csr_data[0].shape[0]:
            slice_data = csr_data[0][start:]
            slice_label = csr_data[1][start]
        else:
            slice_data = csr_data[0][start:start+size]
            slice_label = csr_data[1][start:start+size]
    else:
        if size == -1 or start+size>csr_data[0].shape[0]:
            slice_data=[]
            for field in csr_data[0]:
                slice_data.append(field[start:])
            slice_label = csr_data[1][start:]
        else:
            slice_data = []
            for field in csr_data[0]:
                slice_data.append(field[start:start+size])
            slice_label = slice_data[1][start:start+size]
        return csr_2_input(slice_data),slice_label


def init_var_map(init_vars,init_path = None):
    if init_path is not None:
        load_var_map = pkl.load(init_path)

    var_map = {}
    for var_name,var_shape,init_method,dtype in init_vars:
        if init_method == 'zero':
            var_map[var_name] = tf.Variable(tf.zeros(var_shape,dtype),name=var_name,dtype=dtype)
        elif init_method == 'one':
            var_map[var_name] = tf.Variable(tf.ones(var_shape,dtype = dtype),name=var_name,dtype=dtype)
        elif init_method == 'normal':
            var_map[var_name] = tf.Variable(tf.random_normal(var_shape,mean=0.0,stddev=config.STDDEV,dtype=dtype),name=var_name,dtype=dtype)
        elif init_method == 'tnormal':
            var_map[var_name] = tf.Variable(tf.truncated_normal(var_shape,mean=0.0,stddev=config.STDDEV,dtype=dtype),name=var_name,dtype=dtype)
        elif init_method == 'uniform':
            var_map[var_name] = tf.Variable(tf.random_uniform(var_shape,config.MINVAL,config.MAXVAL),name=var_name,dtype=dtype)
        elif init_method == 'xavier':
            maxval = np.sqrt(6. / np.sum(var_shape))
            minval = -maxval
            var_shape[var_name] = tf.Variable(tf.random_uniform(var_shape,minval,maxval,dtype=dtype),name=var_name,dtype=dtype)
        elif isinstance(init_method,int) or isinstance(init_method,float):
            var_map[var_name] = tf.Variable(tf.ones(var_shape,dtype) * init_method,name=var_name,dtype=dtype)

        elif init_method in load_var_map:
            if load_var_map[init_method].shape == tuple(var_shape):
                var_map[var_name] = tf.Variable(load_var_map[init_method],name=var_name,dtype=dtype)
            else:
                print(f"Badparam: init method {init_method}, shape {var_shape,load_var_map[init_method].shape}")
        else:
            print("BadParam :init method ",init_method)
    return var_map


def activate(wx,activation_fun):
    if activation_fun == 'sigmod':
        return tf.nn.sigmoid(wx)
    elif activation_fun == 'softmax':
        return tf.nn.softmax(wx)
    elif activation_fun == 'relu':
        return tf.nn.relu(wx)
    elif activation_fun == 'tanh':
        return tf.nn.tanh(wx)
    elif activation_fun == 'elu':
        return tf.nn.elu(wx)
    elif activation_fun == 'none':
        return wx
    else :
        return wx


def get_opt(opt_algo,learn_rate,loss):
    if opt_algo == 'adaldeta':
        return tf.train.AdadeltaOptimizer(learn_rate).minimize(loss)
    elif opt_algo == 'adagrad':
        return tf.train.AdagradOptimizer(learn_rate).minimize(loss)
    elif opt_algo == 'adam':
        return tf.train.AdamOptimizer(learn_rate).minimize(loss)
    elif opt_algo == 'ftrl':
        return tf.train.FtrlOptimizer(learn_rate).minimize(loss)
    elif opt_algo == 'gd':
        return tf.train.GradientDescentOptimizer(learn_rate).minimize(loss)
    elif opt_algo == 'padagrad':
        return tf.train.ProximalAdagradOptimizer(learn_rate).minimize(loss)
    elif opt_algo == 'pgd':
        return tf.train.ProximalGradientDescentOptimizer(learn_rate).minimize(loss)
    elif opt_algo == 'rmsprop':
        return tf.train.RMSPropOptimizer(learn_rate).minimize(loss)
    else:
        return tf.train.GradientDescentOptimizer(learn_rate).minimize(loss)



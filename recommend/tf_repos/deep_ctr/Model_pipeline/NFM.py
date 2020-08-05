#!/usr/bin/env python
#coding=utf-8
"""
TensorFlow Implementation of <<Neural Factorization Machines for Sparse Predictive Analytics>> with the fellowing features：
#1 Input pipline using Dataset high level API, Support parallel and prefetch reading
#2 Train pipline using Coustom Estimator by rewriting model_fn
#3 Support distincted training by TF_CONFIG
#4 Support export model for TensorFlow Serving

by lambdaji
"""
#from __future__ import absolute_import
#from __future__ import division
#from __future__ import print_function

#import argparse
import shutil
#import sys
import os
import json
import glob
from datetime import date, timedelta
from time import time
#import gc
#from multiprocessing import Process

#import math
import random
#import pandas as pd
#import numpy as np
import tensorflow as tf

#################### CMD Arguments ####################
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer("dist_mode", 0, "distribuion mode {0-loacal, 1-single_dist, 2-multi_dist}")
tf.app.flags.DEFINE_string("ps_hosts", '', "Comma-separated list of hostname:port pairs")
tf.app.flags.DEFINE_string("worker_hosts", '', "Comma-separated list of hostname:port pairs")
tf.app.flags.DEFINE_string("job_name", '', "One of 'ps', 'worker'")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")
tf.app.flags.DEFINE_integer("num_threads", 8, "Number of threads")
tf.app.flags.DEFINE_integer("feature_size", 117581, "Number of features")
tf.app.flags.DEFINE_integer("field_size", 39, "Number of fields")
tf.app.flags.DEFINE_integer("embedding_size", 64, "Embedding size")
tf.app.flags.DEFINE_integer("num_epochs", 1, "Number of epochs")
tf.app.flags.DEFINE_integer("batch_size", 128, "Number of batch size")
tf.app.flags.DEFINE_integer("log_steps", 1000, "save summary every steps")
tf.app.flags.DEFINE_float("learning_rate", 0.0005, "learning rate")
tf.app.flags.DEFINE_float("l2_reg", 0.001, "L2 regularization")
tf.app.flags.DEFINE_string("loss_type", 'log_loss', "loss type {square_loss, log_loss}")
tf.app.flags.DEFINE_string("optimizer", 'Adam', "optimizer type {Adam, Adagrad, GD, Momentum}")
tf.app.flags.DEFINE_string("deep_layers", '128,64', "deep layers")
tf.app.flags.DEFINE_string("dropout", '0.5,0.8,0.8', "dropout rate")
tf.app.flags.DEFINE_boolean("batch_norm", False, "perform batch normaization (True or False)")
tf.app.flags.DEFINE_float("batch_norm_decay", 0.9, "decay for the moving average(recommend trying decay=0.9)")
tf.app.flags.DEFINE_string("data_dir", '../../data/criteo', "data dir")
tf.app.flags.DEFINE_string("dt_dir", '', "data dt partition")
tf.app.flags.DEFINE_string("model_dir", './model_ckpt/criteo/NFM/', "model check point dir")
tf.app.flags.DEFINE_string("servable_model_dir", '', "export servable model for TensorFlow Serving")
tf.app.flags.DEFINE_string("task_type", 'train', "task type {train, infer, eval, export}")
tf.app.flags.DEFINE_boolean("clear_existing_model", False, "clear existing model or not")

#1 1:0.5 2:0.03519 3:1 4:0.02567 7:0.03708 8:0.01705 9:0.06296 10:0.18185 11:0.02497 12:1 14:0.02565 15:0.03267 17:0.0247 18:0.03158 20:1 22:1 23:0.13169 24:0.02933 27:0.18159 31:0.0177 34:0.02888 38:1 51:1 63:1 132:1 164:1 236:1
def input_fn(filenames, batch_size=32, num_epochs=1, perform_shuffle=False):
    print('Parsing', filenames)
    def decode_libsvm(line):
        #columns = tf.decode_csv(value, record_defaults=CSV_COLUMN_DEFAULTS)
        #features = dict(zip(CSV_COLUMNS, columns))
        #labels = features.pop(LABEL_COLUMN)
        columns = tf.string_split([line], ' ')
        # todo columns= SparseTensor(indices=Tensor("StringSplit:0", shape=(?, 2), dtype=int64), values=Tensor("StringSplit:1", shape=(?,), dtype=string), dense_shape=Tensor("StringSplit:2", shape=(2,), dtype=int64))
        print("columns=",columns)
        labels = tf.string_to_number(columns.values[0], out_type=tf.float32)
        splits = tf.string_split(columns.values[1:], ':')
        # todo splits.dense_shape= Tensor("StringSplit_1:2", shape=(2,), dtype=int64)
        print("splits.dense_shape=",splits.dense_shape)
        id_vals = tf.reshape(splits.values,splits.dense_shape)
        feat_ids, feat_vals = tf.split(id_vals,num_or_size_splits=2,axis=1)
        feat_ids = tf.string_to_number(feat_ids, out_type=tf.int32)
        feat_vals = tf.string_to_number(feat_vals, out_type=tf.float32)
        return {"feat_ids": feat_ids, "feat_vals": feat_vals}, labels

    # Extract lines from input files using the Dataset API, can pass one filename or filename list
    # todo dataset = dataset.map( lambda x :preprocess_for_train(x, image_size, image_size, None)
    #  map转型提供了num_parallel_calls参数指定并行的水平
    #  &
    #  test.txt 的数据格式：
    #  	29	50	5	7260	437	1	4	14		1	0	6	5a9ed9b0	a0e12995	a1e14474	08a40877	25c83c98		964d1fdd	5b392875	a73ee510	de89c3d2	59cd5ae7	8d98db20	8b216f7b	1adce6ef	78c64a1d	3ecdadf7	3486227d	1616f155	21ddcdc9	5840adea	2c277e62		423fab69	54c91918	9b3e8820	e75c9ae9
    # 	27	17	45	28	2	28	27	29	28	1	1		23	68fd1e64	960c983b	9fbfbfd5	38c11726	25c83c98	7e0ccccf	fe06fd10	062b5529	a73ee510	ca53fc84	67360210	895d8bbb	4f8e2224	f862f261	b4cc2435	4c0041e5	e5ba7672	b4abdd09	21ddcdc9	5840adea	36a7ab86		32c7478e	85e4d73f	010f6491	ee63dd9b
    # 	1	1	19	7	1	3	1	7	7	1	1		2	09ca0b81	8947f767	a87e61f7	c4ba2a67	25c83c98	7e0ccccf	ce6020cc	062b5529	a73ee510	b04d3cfe	70dcd184	899eb56b	aca22cf9	b28479f6	a473257f	88f592e4	d4bb7bd8	bd17c3da	1d04f4a4	a458ea53	82bdc0bb		32c7478e	5bdcd9c4	010f6491	cca57dcc
    # 	4	1		6	1051	134	4	35	72	1	1		6	05db9164	532da141	a7ded28e	456b4d8c	25c83c98	fbad5c96	5f29da0e	0b153874	a73ee510	4b344a42	0ad37b4b	8ea37200	f9d99d81	cfef1c29	abd8f51e	9a9902d0	07c540c4	bdc06043			6dfd157c	ad3062eb	423fab69	ef089725
    # 	7	1	25	10	139	74	48	13	44	1	8	2	12	05db9164	207b2d81	2b280564	ad5ffc6b	25c83c98	7e0ccccf	103c17bc	0b153874	a73ee510	8e54038a	e6e0c2dc	2a064dba	e9332a03	07d13a8f	0c67c4ca	7d9b60c8	27c07bd6	395856b0	21ddcdc9	a458ea53	9c3eb598	ad3062eb	3a171ecb	c0b8dfd6	001f3601	7a2fb9af
    # 	8	11	38	9	316	25	8	11	10	1	1		9	05db9164	09e68b86	aa8c1539	85dd697c	25c83c98	7e0ccccf	bc252bd0	5b392875	a73ee510	ef5c0d3c	0bd0c3b3	d8c29807	c0e6befc	8ceecbc8	d2f03b75	c64d548f	e5ba7672	63cdbb21	cf99e5de	5840adea	5f957280		55dd3565	1793a828	e8b83407	b7d9c3bc
    # 	2	1		4	7	4	2	4	4	1	1		4	05db9164	2ae0a573	c5d94b65	5cc8f91d	25c83c98	fe6b92e5	dfc6e241	5b392875	a73ee510	5fe250bc	3547565f	75c79158	12880350	ad1cc976	b046231a	208d4baf	07c540c4	3e340673			6a909d9a		c3dc6cef	1f68c81f
    #  &
    #  tf.data.Dataset.prefetch 的使用 -- http://d0evi1.com/tensorflow/datasets_performance/
    #  为了执行一个training step，你必须首先extract和transform训练数据，接着将它feed给一个运行在加速器上的模型。然而，在一个原始的同步实现（naive synchronous implementation）中，CPU会准备数据，加速器会保持空闲状态。相反地，当加速器在训练模型时，CPU会保持空闲状态。训练过程的时间会是CPU处理时间和加速器训练时间的总和。
    #  Pipeline会将一个training step的预处理和模型执行在时序上重叠。当加速器执行了N个training step时，CPU会为第N+1个step准备数据。这样做是为了减少总时间。
    #  如果没有做pipeling，CPU和GPU/TPU的空闲时间会有很多.
    #  tf.data API通过tf.data.Dataset.prefetch转换，提供了一个软件实现的管道机制(software pipeling），该转换可以被用于将数据生产时间和数据消费时间相解耦。特别的，该转换会使用一个后台线程，以及一个内部缓存（internal buffer），来prefetch来自input dataset的元素（在它们被请求之前）。这样，为了达到上述的pipeline效果，你可以添加prefetch(1) 作为最终转换到你的dataset pipeline中（或者prefetch(n)，如果单个training step消费n个元素的话）
    dataset = tf.data.TextLineDataset(filenames).map(decode_libsvm, num_parallel_calls=10).prefetch(500000)    # multi-thread pre-process then prefetch
    # todo dataset=<DatasetV1Adapter shapes: ({feat_ids: (?, ?), feat_vals: (?, ?)}, ()), types: ({feat_ids: tf.int32, feat_vals: tf.float32}, tf.float32)>
    #  <class 'tensorflow.python.data.ops.dataset_ops.DatasetV1Adapter'>
    # Randomizes input using a window of 256 elements (read into memory)
    if perform_shuffle:
    # todo
    #  dataset = dataset.shuffle(buffer_size):buffle的机制是在内存缓冲区中保存一个buffer_size条数据，每读入一条数据后，
    #  从这个缓冲区中随机选择一条数据进行输出，缓冲区的大小越大，随机的性能就越好，但是也更耗费内存。
    #  https://www.jianshu.com/p/eba6841a0ff7
        dataset = dataset.shuffle(buffer_size=256)

    # epochs from blending together.
    # todo dataset = dataset.repeat(N) 表示将数据复制N份
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size) # Batch size to use

    # todo 语句iterator = dataset.make_one_shot_iterator()从dataset中实例化了一个Iterator，这个Iterator是一个“one shot iterator”，即只能从头到尾读取一次。
    #  one_element = iterator.get_next()表示从iterator里取出一个元素。由于这是非Eager模式，所以one_element只是一个Tensor，并不是一个实际的值。调用sess.run(one_element)后，才能真正地取出一个值。
    #  如果一个dataset中元素被读取完了，再尝试sess.run(one_element)的话，就会抛出tf.errors.OutOfRangeError异常，这个行为与使用队列方式读取数据的行为是一致的。
    iterator = dataset.make_one_shot_iterator()
    batch_features, batch_labels = iterator.get_next()
    #return tf.reshape(batch_ids,shape=[-1,field_size]), tf.reshape(batch_vals,shape=[-1,field_size]), batch_labels
    return batch_features, batch_labels

def model_fn(features, labels, mode, params):
    """Bulid Model function f(x) for Estimator."""
    #------hyperparameters----
    field_size = params["field_size"]
    feature_size = params["feature_size"]
    embedding_size = params["embedding_size"]
    l2_reg = params["l2_reg"]
    learning_rate = params["learning_rate"]
    #optimizer = params["optimizer"]
    layers = list(map(int, params["deep_layers"].split(',')))
    # print(f'params["deep_layers"]={params["deep_layers"]}; layers={list(layers),list(layers)}')
    # print(f"layers==={list(layers)}")
    # print(f"layers={layers, layers, layers}")
    dropout = list(map(float, params["dropout"].split(',')))

    #------bulid weights------
    Global_Bias = tf.get_variable(name='bias', shape=[1], initializer=tf.constant_initializer(0.0))
    Feat_Bias = tf.get_variable(name='linear', shape=[feature_size], initializer=tf.glorot_normal_initializer())
    Feat_Emb = tf.get_variable(name='emb', shape=[feature_size,embedding_size], initializer=tf.glorot_normal_initializer())

    #------build feaure-------
    feat_ids  = features['feat_ids']
    # todo feat_ids== Tensor("IteratorGetNext:0", shape=(?, ?, ?), dtype=int32, device=/device:CPU:0)
    # print("feat_ids==",feat_ids)
    feat_ids = tf.reshape(feat_ids,shape=[-1,field_size])
    feat_vals = features['feat_vals']
    feat_vals = tf.reshape(feat_vals,shape=[-1,field_size])

    #------build f(x)------
    with tf.variable_scope("Linear-part"):
        feat_wgts = tf.nn.embedding_lookup(Feat_Bias, feat_ids) 		# None * F * 1
        y_linear = tf.reduce_sum(tf.multiply(feat_wgts, feat_vals),1)
        # todo y_linear =  Tensor("Linear-part/Sum:0", shape=(?,), dtype=float32)
        #  tf.multiply(feat_wgts, feat_vals)= Tensor("Linear-part/Mul_1:0", shape=(?, 39), dtype=float32)
        #  feat_wgts= Tensor("Linear-part/embedding_lookup:0", shape=(?, 39), dtype=float32)
        #  feat_vals= Tensor("Reshape_1:0", shape=(?, 39), dtype=float32)
        #  这里非线性项输出的一个数吗？
        # print("y_linear = ",y_linear,'tf.multiply(feat_wgts, feat_vals)=',tf.multiply(feat_wgts, feat_vals),
        #       "feat_wgts=",feat_wgts, "feat_vals=",feat_vals)

    with tf.variable_scope("BiInter-part"):
        embeddings = tf.nn.embedding_lookup(Feat_Emb, feat_ids) 		# None * F * K
        feat_vals = tf.reshape(feat_vals, shape=[-1, field_size, 1])
        # todo embeddings1 = Tensor("BiInter-part/embedding_lookup:0", shape=(?, 39, 64), dtype=float32)
        #  embeddings2 = Tensor("BiInter-part/Mul:0", shape=(?, 39, 64), dtype=float32)
        #  sum_square_emb = Tensor("BiInter-part/Square:0", shape=(?, 64), dtype=float32)
        #  square_sum_emb = Tensor("BiInter-part/Sum_1:0", shape=(?, 64), dtype=float32)
        print("embeddings1=",embeddings)
        embeddings = tf.multiply(embeddings, feat_vals) 				# vij * xi
        print("embeddings2=",embeddings)
        sum_square_emb = tf.square(tf.reduce_sum(embeddings,1))
        print("sum_square_emb=",sum_square_emb)
        square_sum_emb = tf.reduce_sum(tf.square(embeddings),1)
        print("square_sum_emb=",square_sum_emb)
        deep_inputs = 0.5*tf.subtract(sum_square_emb, square_sum_emb)	# None * K

    with tf.variable_scope("Deep-part"):
        if mode == tf.estimator.ModeKeys.TRAIN:
            train_phase = True
        else:
            train_phase = False

        if mode == tf.estimator.ModeKeys.TRAIN:
            deep_inputs = tf.nn.dropout(deep_inputs, keep_prob=dropout[0]) 						# None * K
        # print(f"layers={list(layers)}")
        for i in range(len(layers)):
            deep_inputs = tf.contrib.layers.fully_connected(inputs=deep_inputs, num_outputs=layers[i], \
                weights_regularizer=tf.contrib.layers.l2_regularizer(l2_reg), scope='mlp%d' % i)

            if FLAGS.batch_norm:
                deep_inputs = batch_norm_layer(deep_inputs, train_phase=train_phase, scope_bn='bn_%d' %i)   #放在RELU之后 https://github.com/ducha-aiki/caffenet-benchmark/blob/master/batchnorm.md#bn----before-or-after-relu
            if mode == tf.estimator.ModeKeys.TRAIN:
                deep_inputs = tf.nn.dropout(deep_inputs, keep_prob=dropout[i])                              #Apply Dropout after all BN layers and set dropout=0.8(drop_ratio=0.2)
                #deep_inputs = tf.layers.dropout(inputs=deep_inputs, rate=dropout[i], training=mode == tf.estimator.ModeKeys.TRAIN)

        # todo 这里全连接层的输出是一个数值，而非向量
        y_deep = tf.contrib.layers.fully_connected(inputs=deep_inputs, num_outputs=1, activation_fn=tf.identity, \
            weights_regularizer=tf.contrib.layers.l2_regularizer(l2_reg), scope='deep_out')
        y_d = tf.reshape(y_deep,shape=[-1])

    with tf.variable_scope("NFM-out"):
        #y_bias = Global_Bias * tf.ones_like(labels, dtype=tf.float32)  # None * 1  warning;这里不能用label，否则调用predict/export函数会出错，train/evaluate正常；初步判断estimator做了优化，用不到label时不传
        y_bias = Global_Bias * tf.ones_like(y_d, dtype=tf.float32)     	# None * 1
        y = y_bias + y_linear + y_d
        pred = tf.sigmoid(y)
    # todo y_deep= Tensor("Deep-part/deep_out/Identity:0", shape=(?, 1), dtype=float32)
    #  y_d= Tensor("Deep-part/Reshape:0", shape=(?,), dtype=float32)
    #  Global_Bias= <tf.Variable 'bias:0' shape=(1,) dtype=float32_ref>
    #  y_bias Tensor("NFM-out/mul:0", shape=(?,), dtype=float32)
    #  y_linear= Tensor("Linear-part/Sum:0", shape=(?,), dtype=float32)
    # print("y_deep=",y_deep,"y_d=",y_d,"Global_Bias=",Global_Bias,"y_bias",y_bias,"y_linear=",y_linear)

    predictions={"prob": pred}
    export_outputs = {tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: tf.estimator.export.PredictOutput(predictions)}
    # Provide an estimator spec for `ModeKeys.PREDICT`
    if mode == tf.estimator.ModeKeys.PREDICT:
        # todo tf.estimator.EstimatorSpec -- https://blog.csdn.net/qq_32806793/article/details/85010302
        #  主要参数说明
        #  eval_metric_ops：Dict of metric results keyed by name.
        #  predictions: Predictions Tensor or dict of Tensor.(模型的预测输出，主要是在infer阶段，在分类是：预测的类别，在文本生成是：生成的文本)
        #  loss：Training loss Tensor. 损失。主要用在train 和 dev中
        #  train_op ：Op for the training step.（是一个操作，用来训练）
        return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=predictions,
                export_outputs=export_outputs)

    #------bulid loss------
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y, labels=labels)) + \
        l2_reg * tf.nn.l2_loss(Feat_Bias) + l2_reg * tf.nn.l2_loss(Feat_Emb)

    # Provide an estimator spec for `ModeKeys.EVAL`
    eval_metric_ops = {
        "auc": tf.metrics.auc(labels, pred)
    }
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=predictions,
                loss=loss,
                eval_metric_ops=eval_metric_ops)

    #------bulid optimizer------
    if FLAGS.optimizer == 'Adam':
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8)
    elif FLAGS.optimizer == 'Adagrad':
        optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate, initial_accumulator_value=1e-8)
    elif FLAGS.optimizer == 'Momentum':
        optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.95)
    elif FLAGS.optimizer == 'ftrl':
        optimizer = tf.train.FtrlOptimizer(learning_rate)

    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

    # Provide an estimator spec for `ModeKeys.TRAIN` modes
    if mode == tf.estimator.ModeKeys.TRAIN:
        return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=predictions,
                loss=loss,
                train_op=train_op)

    # Provide an estimator spec for `ModeKeys.EVAL` and `ModeKeys.TRAIN` modes.
    #return tf.estimator.EstimatorSpec(
    #        mode=mode,
    #        loss=loss,
    #        train_op=train_op,
    #        predictions={"prob": pred},
    #        eval_metric_ops=eval_metric_ops)

def batch_norm_layer(x, train_phase, scope_bn):
    bn_train = tf.contrib.layers.batch_norm(x, decay=FLAGS.batch_norm_decay, center=True, scale=True, updates_collections=None, is_training=True,  reuse=None, scope=scope_bn)
    bn_infer = tf.contrib.layers.batch_norm(x, decay=FLAGS.batch_norm_decay, center=True, scale=True, updates_collections=None, is_training=False, reuse=True, scope=scope_bn)
    z = tf.cond(tf.cast(train_phase, tf.bool), lambda: bn_train, lambda: bn_infer)
    return z

def set_dist_env():
    # todo FLAGS.dist_mode= 0
    print("FLAGS.dist_mode=",FLAGS.dist_mode)
    if FLAGS.dist_mode == 1:        # 本地分布式测试模式1 chief, 1 ps, 1 evaluator
        ps_hosts = FLAGS.ps_hosts.split(',')
        chief_hosts = FLAGS.chief_hosts.split(',')
        task_index = FLAGS.task_index
        job_name = FLAGS.job_name
        print('ps_host', ps_hosts)
        print('chief_hosts', chief_hosts)
        print('job_name', job_name)
        print('task_index', str(task_index))
        # 无worker参数
        tf_config = {
            'cluster': {'chief': chief_hosts, 'ps': ps_hosts},
            'task': {'type': job_name, 'index': task_index }
        }
        print(json.dumps(tf_config))
        os.environ['TF_CONFIG'] = json.dumps(tf_config)
    elif FLAGS.dist_mode == 2:      # 集群分布式模式
        ps_hosts = FLAGS.ps_hosts.split(',')
        worker_hosts = FLAGS.worker_hosts.split(',')
        chief_hosts = worker_hosts[0:1] # get first worker as chief
        worker_hosts = worker_hosts[2:] # the rest as worker
        task_index = FLAGS.task_index
        job_name = FLAGS.job_name
        print('ps_host', ps_hosts)
        print('worker_host', worker_hosts)
        print('chief_hosts', chief_hosts)
        print('job_name', job_name)
        print('task_index', str(task_index))
        # use #worker=0 as chief
        if job_name == "worker" and task_index == 0:
            job_name = "chief"
        # use #worker=1 as evaluator
        if job_name == "worker" and task_index == 1:
            job_name = 'evaluator'
            task_index = 0
        # the others as worker
        if job_name == "worker" and task_index > 1:
            task_index -= 2

        tf_config = {
            'cluster': {'chief': chief_hosts, 'worker': worker_hosts, 'ps': ps_hosts},
            'task': {'type': job_name, 'index': task_index }
        }
        print(json.dumps(tf_config))
        os.environ['TF_CONFIG'] = json.dumps(tf_config)

def main(_):
    #------check Arguments------
    if FLAGS.dt_dir == "":
        FLAGS.dt_dir = (date.today() + timedelta(-1)).strftime('%Y%m%d')
    FLAGS.model_dir = FLAGS.model_dir + FLAGS.dt_dir
    #FLAGS.data_dir  = FLAGS.data_dir + FLAGS.dt_dir

    print('task_type ', FLAGS.task_type)
    print('model_dir ', FLAGS.model_dir)
    print('data_dir ', FLAGS.data_dir)
    print('dt_dir ', FLAGS.dt_dir)
    print('num_epochs ', FLAGS.num_epochs)
    print('feature_size ', FLAGS.feature_size)
    print('field_size ', FLAGS.field_size)
    print('embedding_size ', FLAGS.embedding_size)
    print('batch_size ', FLAGS.batch_size)
    print('deep_layers ', FLAGS.deep_layers)
    print('dropout ', FLAGS.dropout)
    print('loss_type ', FLAGS.loss_type)
    print('optimizer ', FLAGS.optimizer)
    print('learning_rate ', FLAGS.learning_rate)
    print('l2_reg ', FLAGS.l2_reg)

    #------init Envs------
    # todo 返回所有匹配的文件路径列表。
    # 	tr_files: ['../../data/criteo/tr.libsvm']
    # 	va_files: ['../../data/criteo/va.libsvm']
    # 	te_files: ['../../data/criteo/te.libsvm']
    tr_files = glob.glob("%s/tr*libsvm" % FLAGS.data_dir)
    random.shuffle(tr_files)
    print("tr_files:", tr_files)
    va_files = glob.glob("%s/va*libsvm" % FLAGS.data_dir)
    print("va_files:", va_files)
    te_files = glob.glob("%s/te*libsvm" % FLAGS.data_dir)
    print("te_files:", te_files)

    if FLAGS.clear_existing_model:
        try:
            shutil.rmtree(FLAGS.model_dir)
        except Exception as e:
            print(e, "at clear_existing_model")
        else:
            print("existing model cleaned at %s" % FLAGS.model_dir)

    set_dist_env()

    #------bulid Tasks------
    model_params = {
        "field_size": FLAGS.field_size,
        "feature_size": FLAGS.feature_size,
        "embedding_size": FLAGS.embedding_size,
        "learning_rate": FLAGS.learning_rate,
        "l2_reg": FLAGS.l2_reg,
        "deep_layers": FLAGS.deep_layers,
        "dropout": FLAGS.dropout
    }
    # config = tf.estimator.RunConfig().replace(session_config = tf.ConfigProto(device_count={'GPU':0, 'CPU':FLAGS.num_threads}),
    #         log_step_count_steps=FLAGS.log_steps, save_summary_steps=FLAGS.log_steps)
    config = tf.estimator.RunConfig().replace(
        session_config = tf.ConfigProto(device_count={'CPU':FLAGS.num_threads}),
        log_step_count_steps=FLAGS.log_steps, save_summary_steps=FLAGS.log_steps)
    Estimator = tf.estimator.Estimator(model_fn=model_fn, model_dir=FLAGS.model_dir, params=model_params, config=config)

    # todo tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
    #  其中 estimator 是一个 tf.estimator.Estimator 对象，用于指定模型函数以及其它相关参数；
    #  train_spec 是一个 tf.estimator.TrainSpec 对象，用于指定训练的输入函数以及其它参数；
    #  eval_spec 是一个 tf.estimator.EvalSpec 对象，用于指定验证的输入函数以及其它参数。
    #  &
    #  tf.estimator.Estimator(model_fn, model_dir=None, config=None,params=None, warm_start_from=None)
    #  其中 model_fn 是模型函数；model_dir 是训练时模型保存的路径；config 是 tf.estimator.RunConfig 的配置对象；
    #  params 是传入 model_fn 的超参数字典；warm_start_from 或者是一个预训练文件的路径，或者是一个 tf.estimator.WarmStartSettings 对象，用于完整的配置热启动参数。
    #  &
    #  tf.estimator.TrainSpec(input_fn, max_steps, hooks)
    #  其中 input_fn 用来提供训练时的输入数据；max_steps 指定总共训练多少步；hooks 是一个 tf.train.SessionRunHook 对象，用来配置分布式训练等参数。
    #  &
    #  tf.estimator.EvalSpec(input_fn,steps=100,name=None,hooks=None,exporters=None,start_delay_secs=120,throttle_secs=600)
    #  其中 input_fn 用来提供验证时的输入数据；steps 指定总共验证多少步（一般设定为 None 即可）；hooks 用来配置分布式训练等参数；
    #  exporters 是一个 Exporter 迭代器，会参与到每次的模型验证；start_delay_secs 指定多少秒之后开始模型验证；
    #  throttle_secs 指定多少秒之后重新开始新一轮模型验证（当然，如果没有新的模型断点保存，则该数值秒之后不会进行模型验证，因此这是新一轮模型验证需要等待的最小秒数）。
    #  &
    #  定义模型函数 model_fn，返回类 tf.estimator.EstimatorSpec 的一个实例。
    #  def create_model_fn(features, labels, mode, params=None):
    #     pass
    #     return tf.estimator.EstimatorSpec(mode=mode,predictions=prediction_dict,loss=loss,train_op=train_op,...)
    #  其中 features，labels 可以是一个张量，也可以是由张量组成的一个字典；mode 指定训练模式，可以取 （TRAIN, EVAL, PREDICT）三者之一；params 是一个（可要可不要的）字典，指定其它超参数。
    #  model_fn 必须定义模型的预测结果、损失、优化器等，它返回类 tf.estimator.EstimatorSpec 的一个对象。
    #  &
    #  tf.estimator.EstimatorSpec(mode,predictions=None,loss=None,train_op=None,eval_metric_ops=None,export_outputs=None,training_chief_hooks=None,training_hooks=None,scaffold=None,evaluation_hooks=None,prediction_hooks=None)
    #  其中 mode 指定当前是处于训练、验证还是预测状态；predictions 是预测的一个张量，或者是由张量组成的一个字典；loss 是损失张量；train_op 指定优化操作；
    #  eval_metric_ops 指定各种评估度量的字典，这个字典的值必须是如下两种形式：Metric 类的实例；调用某个评估度量函数的结果对 (metric_tensor, update_op)；
    #  参数 export_outputs 只用于模型保存，描述了导出到 SavedModel 的输出格式；参数 scaffold 是一个 tf.train.Scaffold 对象，可以在训练阶段初始化、保存等时使用。
    #  &
    #  定义输入函数 input_fn，返回如下两种格式之一：
    #  tf.data.Dataset 对象：这个对象的输出必须是元组队 (features, labels)，而且必须满足下一条返回格式的同等约束；
    #  元组 (features, labels)：features 以及 labels 都必须是一个张量或由张量组成的字典。
    #  &
    #  https://www.jianshu.com/p/b8930fa13ea7
    #  &
    #  https://www.cnblogs.com/zongfa/p/10149483.html
    if FLAGS.task_type == 'train':
        train_spec = tf.estimator.TrainSpec(input_fn=lambda: input_fn(tr_files, num_epochs=FLAGS.num_epochs, batch_size=FLAGS.batch_size))
        eval_spec = tf.estimator.EvalSpec(input_fn=lambda: input_fn(va_files, num_epochs=1, batch_size=FLAGS.batch_size), steps=None, start_delay_secs=1000, throttle_secs=1200)
        tf.estimator.train_and_evaluate(Estimator, train_spec, eval_spec)

    elif FLAGS.task_type == 'eval':
        Estimator.evaluate(input_fn=lambda: input_fn(va_files, num_epochs=1, batch_size=FLAGS.batch_size))
    # todo 这里 infer 是什么意思？
    elif FLAGS.task_type == 'infer':
        preds = Estimator.predict(input_fn=lambda: input_fn(te_files, num_epochs=1, batch_size=FLAGS.batch_size), predict_keys="prob")
        with open(FLAGS.data_dir+"/pred.txt", "w") as fo:
            for prob in preds:
                fo.write("%f\n" % (prob['prob']))
    elif FLAGS.task_type == 'export':
        #feature_spec = tf.feature_column.make_parse_example_spec(feature_columns)
        #feature_spec = {
        #    'feat_ids': tf.FixedLenFeature(dtype=tf.int64, shape=[None, FLAGS.field_size]),
        #    'feat_vals': tf.FixedLenFeature(dtype=tf.float32, shape=[None, FLAGS.field_size])
        #}
        #serving_input_receiver_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec)
        feature_spec = {
            'feat_ids': tf.placeholder(dtype=tf.int64, shape=[None, FLAGS.field_size], name='feat_ids'),
            'feat_vals': tf.placeholder(dtype=tf.float32, shape=[None, FLAGS.field_size], name='feat_vals')
        }
        serving_input_receiver_fn = tf.estimator.export.build_raw_serving_input_receiver_fn(feature_spec)
        # todo export_savedmodel可将训练好的模型导出成SavedModel格式
        #  其中serving_input_receiver_fn表示tensorflow serving接收请求输入(并做相应处理)的函数，该函数返回一个 tf.estimator.export.ServingInputReceiver 对象。
        #  serving_input_receiver_fn可自定义，也可直接利用tensorflow封装好的API(tf.estimator.export.build_parsing_serving_input_receiver_fn或tf.estimator.export.build_raw_serving_input_receiver_fn)。
        #  build_parsing_serving_input_receiver_fn用于接收序列化的tf.Examples。而build_raw_serving_input_receiver_fn用于接收原生的Tensor。
        #  这里只传入了一个 容器怎么就可以定义模型了？或者说这里加载的模型是什么？
        Estimator.export_savedmodel(FLAGS.servable_model_dir, serving_input_receiver_fn)
    print("model over")

if __name__ == "__main__":
    # todo 在tensorflow中函数可以直接log打印
    #  tensorflow使用5个不同级别的日志消息。按照上升的顺序，他们是debug，info,warn,error和fatal。在任何级别配置日志记录时，tf将输出与该级别相对用的所有日志信息。
    #  在跟踪模型训练时，将级别调整为info，将提供适合操作正在进行的其他反馈。比如可以在tensorboard中看训练过程的loss，acc的变化情况。
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
    # df_final[df_final['是否同期多单'] == ' ']

    import xgboost as xgb

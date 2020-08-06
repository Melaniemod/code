#!/usr/bin/env python
#coding=utf-8
"""
TensorFlow Implementation of <<Attentional Factorization Machines: Learning the Weight of Feature Interactions via Attention Networks>>
with the fellowing features：
#1 Input pipline using Dataset high level API, Support parallel and prefetch reading
#2 Train pipline using Coustom Estimator by rewriting model_fn
#3 Support distincted training by TF_CONFIG
#4 Support export model for online predicting service using TensorFlow Serving

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
tf.app.flags.DEFINE_integer("embedding_size", 256, "Embedding size")
tf.app.flags.DEFINE_integer("num_epochs", 1, "Number of epochs")
tf.app.flags.DEFINE_integer("batch_size", 128, "Number of batch size")
tf.app.flags.DEFINE_integer("log_steps", 1000, "save summary every steps")
tf.app.flags.DEFINE_float("learning_rate", 0.1, "learning rate")
tf.app.flags.DEFINE_float("l2_reg", 1.0, "L2 regularization")
tf.app.flags.DEFINE_string("loss_type", 'log_loss', "loss type {square_loss, log_loss}")
tf.app.flags.DEFINE_string("optimizer", 'Adam', "optimizer type {Adam, Adagrad, GD, Momentum}")
tf.app.flags.DEFINE_string("attention_layers", '256', "Attention Net mlp layers")
tf.app.flags.DEFINE_string("dropout", '1.0,0.5', "dropout rate")
tf.app.flags.DEFINE_string("data_dir", '../../data/criteo/', "data dir")
tf.app.flags.DEFINE_string("dt_dir", '', "data dt partition")
tf.app.flags.DEFINE_string("model_dir", './model_ckpt/criteo/AFM', "model check point dir")
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
        # todo tf.string_split函数
        #  tf.string_split(
        #      source,
        #      delimiter=' ',
        #      skip_empty=True
        #  )
        #  '''
        #  @函数意义：将基于 delimiter 的 source 的元素拆分为 SparseTensor
        #  @source：需要操作的对象，一般是[字符串或者多个字符串]构成的列表；---注意是列表哦！！！
        #  @delimiter:分割符,默认空字符串
        #  @skip_empty：默认True，暂时没用到过
        #  '''
        #  demo
        #  #  当对象是一个字符串
        #  a = 'we do it'
        #  tf.string_split([a])
        #  #  返回值如下
        #  SparseTensorValue(indices=array([[0, 0],[0, 1],[0, 2]]),
        #                     values=array(['we', 'do', 'it'], dtype=object),
        #                       dense_shape=array([1, 3]))
        #  #  当对象是多个字符串
        #  b = 'we can do it'
        #  c = [a,b]
        #  tf.string_split(c)
        #  #  返回值如下
        #  SparseTensorValue(indices=array([[0, 0],
        #         [0, 1],
        #         [0, 2],
        #         [1, 0],
        #         [1, 1],
        #         [1, 2],
        #         [1, 3]], dtype=int64), values=array(['we', 'do', 'it', 'we', 'can', 'do', 'it'], dtype=object), dense_shape=array([2, 4], dtype=int64))
        #  可以看到几个要点：
        #  1.传入的元素是字符串，但是必须是列表包括进去，不然会报格式错误！
        #  2.返回了稀疏矩阵(SparseTensorValue)的下标(indices)，和值(value),以及类型，和输入数据的维度(dense_shape)
        #  3.到这一步已经很明显了，这个函数有split()的作用，可以从value获取我们要的东西。
        #  返回值有三个参数，一个是indices,一个是values,一个是dense_shape.
        columns = tf.string_split([line], ' ')
        labels = tf.string_to_number(columns.values[0], out_type=tf.float32)
        splits = tf.string_split(columns.values[1:], ':')
        # todo 这里没有index就直接reshape成
        id_vals = tf.reshape(splits.values,splits.dense_shape)
        # todo splits= SparseTensor(indices=Tensor("StringSplit_1:0", shape=(?, 2), dtype=int64), values=Tensor("StringSplit_1:1", shape=(?,), dtype=string), dense_shape=Tensor("StringSplit_1:2", shape=(2,), dtype=int64))
        #  Tensor("StringSplit_1:1", shape=(?,), dtype=string)
        #  Tensor("StringSplit_1:2", shape=(2,), dtype=int64)
        #  Tensor("Reshape:0", shape=(?, ?), dtype=string)
        # print("splits=",splits,splits.values,splits.dense_shape,id_vals)
        # todo  tf.split(value,num_or_size_splits,axis=0,num=None,name='split')
        #  https://blog.csdn.net/mls0311/article/details/82052472
        #  value：准备切分的张量
        #  num_or_size_splits：准备切成几份
        #  axis : 准备在第几个维度上进行切割
        #  其中分割方式分为两种
        #  1. 如果num_or_size_splits 传入的 是一个整数，那直接在axis=D这个维度上把张量平均切分成几个小张量
        #  2. 如果num_or_size_splits 传入的是一个向量（这里向量各个元素的和要跟原本这个维度的数值相等）就根据这个向量有几个元素分为几项）
        feat_ids, feat_vals = tf.split(id_vals,num_or_size_splits=2,axis=1)
        feat_ids = tf.string_to_number(feat_ids, out_type=tf.int32)
        feat_vals = tf.string_to_number(feat_vals, out_type=tf.float32)
        #feat_ids = tf.reshape(feat_ids,shape=[-1,FLAGS.field_size])
        #for i in range(splits.dense_shape.eval()[0]):
        #    feat_ids.append(tf.string_to_number(splits.values[2*i], out_type=tf.int32))
        #    feat_vals.append(tf.string_to_number(splits.values[2*i+1]))
        #return tf.reshape(feat_ids,shape=[-1,field_size]), tf.reshape(feat_vals,shape=[-1,field_size]), labels
        return {"feat_ids": feat_ids, "feat_vals": feat_vals}, labels

    # Extract lines from input files using the Dataset API, can pass one filename or filename list
    dataset = tf.data.TextLineDataset(filenames).map(decode_libsvm, num_parallel_calls=10).prefetch(500000)    # multi-thread pre-process then prefetch
    print("dataset ---")
    # Randomizes input using a window of 256 elements (read into memory)
    if perform_shuffle:
        dataset = dataset.shuffle(buffer_size=256)

    # epochs from blending together.
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size) # Batch size to use

    #return dataset.make_one_shot_iterator()
    iterator = dataset.make_one_shot_iterator()
    batch_features, batch_labels = iterator.get_next()
    #return tf.reshape(batch_ids,shape=[-1,field_size]), tf.reshape(batch_vals,shape=[-1,field_size]), batch_labels
    return batch_features, batch_labels

def model_fn(features, labels, mode, params):
    """Bulid Model function f(x) for Estimator."""
    # todo  mode是如何传过去的呢？
    #------hyperparameters----
    # todo mode,ModeKeys eval train eval infer
    #  为什么 task_type = train的时候，mode 也是eval???
    print("mode,ModeKeys",mode,tf.estimator.ModeKeys.TRAIN,tf.estimator.ModeKeys.EVAL,tf.estimator.ModeKeys.PREDICT)
    field_size = params["field_size"]
    feature_size = params["feature_size"]
    embedding_size = params["embedding_size"]
    l2_reg = params["l2_reg"]
    learning_rate = params["learning_rate"]
    #optimizer = params["optimizer"]
    layers = list(map(int, params["attention_layers"].split(',')))
    dropout = list(map(float, params["dropout"].split(',')))

    #------bulid weights------
    Global_Bias = tf.get_variable(name='bias', shape=[1], initializer=tf.constant_initializer(0.0))
    Feat_Bias = tf.get_variable(name='linear', shape=[feature_size], initializer=tf.glorot_normal_initializer())
    Feat_Emb = tf.get_variable(name='emb', shape=[feature_size,embedding_size], initializer=tf.glorot_normal_initializer())

    #------build feaure-------
    feat_ids  = features['feat_ids']
    feat_ids = tf.reshape(feat_ids,shape=[-1,field_size])
    feat_vals = features['feat_vals']
    feat_vals = tf.reshape(feat_vals,shape=[-1,field_size])

    #------build f(x)------
    with tf.variable_scope("Linear-part"):
        feat_wgts = tf.nn.embedding_lookup(Feat_Bias, feat_ids) # None * F * 1
        y_linear = tf.reduce_sum(tf.multiply(feat_wgts, feat_vals),1)

    with tf.variable_scope("Pairwise-Interaction-Layer"):
        embeddings = tf.nn.embedding_lookup(Feat_Emb, feat_ids) # None * F * K
        feat_vals = tf.reshape(feat_vals, shape=[-1, field_size, 1])
        embeddings = tf.multiply(embeddings, feat_vals) #vij*xi      None * F * K
        # todo Tensor("Pairwise-Interaction-Layer/Mul:0", shape=(?, 39, 256), dtype=float32)
        # print("Pairwise-Interaction-Layer-embeddings",embeddings)

        num_interactions = int(field_size*(field_size-1)/2)
        element_wise_product_list = []
        for i in range(0, field_size):
            for j in range(i+1, field_size):
                element_wise_product_list.append(tf.multiply(embeddings[:,i,:], embeddings[:,j,:]))
        # todo tf.stack其作用类似于tf.concat，都是拼接两个张量，而不同之处在于，tf.concat拼接的是除了拼接维度axis外其他维度的shape完全相同的张量，
        #  并且产生的张量的阶数不会发生变化，而tf.stack则会在新的张量阶上拼接，产生的张量的阶数将会增加，
        element_wise_product = tf.stack(element_wise_product_list) 								# (F*(F-1)) * None * K
        # todo Tensor("Pairwise-Interaction-Layer/stack:0", shape=(741, ?, 256), dtype=float32)
        # print("element_wise_product",element_wise_product)
        element_wise_product = tf.transpose(element_wise_product, perm=[1,0,2]) 				# None * (F*(F-1)) * K
        #interactions = tf.reduce_sum(element_wise_product, 2, name="interactions")

    with tf.variable_scope("Attention-part"):
        deep_inputs = tf.reshape(element_wise_product, shape=[-1, embedding_size]) 				# (None * (F*(F-1))) * K
        for i in range(len(layers)):
            deep_inputs = tf.contrib.layers.fully_connected(inputs=deep_inputs, num_outputs=layers[i], \
                weights_regularizer=tf.contrib.layers.l2_regularizer(l2_reg), scope='mlp%d' % i)
        # todo deep_inputs= Tensor("Attention-part/mlp0/Relu:0", shape=(?, 256), dtype=float32)
        # print("deep_inputs=",deep_inputs)
        aij = tf.contrib.layers.fully_connected(inputs=deep_inputs, num_outputs=1, activation_fn=tf.identity, \
            weights_regularizer=tf.contrib.layers.l2_regularizer(l2_reg), scope='attention_out')# (None * (F*(F-1))) * 1
        # todo aij= Tensor("Attention-part/attention_out/Identity:0", shape=(?, 1), dtype=float32)
        # print('aij=',aij)

        #aij_reshape = tf.reshape(aij, shape=[-1, num_interactions, 1])							# None * (F*(F-1)) * 1
        aij_softmax = tf.nn.softmax(tf.reshape(aij, shape=[-1, num_interactions, 1]), dim=1, name='attention_soft')
        if mode == tf.estimator.ModeKeys.TRAIN:
            aij_softmax = tf.nn.dropout(aij_softmax, keep_prob=dropout[0])

    with tf.variable_scope("Attention-based-Pooling"):
        y_emb = tf.reduce_sum(tf.multiply(aij_softmax, element_wise_product), 1) 				# None * K
        if mode == tf.estimator.ModeKeys.TRAIN:
            y_emb = tf.nn.dropout(y_emb, keep_prob=dropout[1])

        y_d = tf.contrib.layers.fully_connected(inputs=y_emb, num_outputs=1, activation_fn=tf.identity, \
            weights_regularizer=tf.contrib.layers.l2_regularizer(l2_reg), scope='deep_out')		# None * 1
        y_deep = tf.reshape(y_d,shape=[-1])

    with tf.variable_scope("AFM-out"):
        #y_bias = Global_Bias * tf.ones_like(labels, dtype=tf.float32)  # None * 1  warning;这里不能用label，否则调用predict/export函数会出错，train/evaluate正常；初步判断estimator做了优化，用不到label时不传
        y_bias = Global_Bias * tf.ones_like(y_deep, dtype=tf.float32)   # None * 1
        y = y_bias + y_linear + y_deep
        pred = tf.sigmoid(y)

    predictions={"prob": pred}
    # todo   TensorFlow Serving服务框架     -- http://octopuscoder.github.io/2019/05/07/%E4%BD%BF%E7%94%A8TensorFlow-Serving%E5%BF%AB%E9%80%9F%E9%83%A8%E7%BD%B2%E6%A8%A1%E5%9E%8B/
    #   框架分为模型训练、模型上线和服务使用三部分。模型训练与正常的训练过程一致，只是导出时需要按照TF Serving的标准定义输入、输出和签名。
    #   模型上线时指定端口号和模型路径后，通过tensorflow_model_server命令启动服务。服务使用可通过grpc和RESTfull方式请求。
    #   &
    #   模型导出时，需指定模型的输入和输出，并在tags中包含”serve”，在实际使用中，TF Serving要求导出模型包含”serve”这个tag。
    #   此外，还需要指定默认签名，tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY = “serving_default”，
    #   此外tf.saved_model.signature_constants定义了三类签名，分别是：分类classify,回归regress,预测predict

    #  todo 一个head必须使用signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY进行命名，
    #   它表示当一个inference请求没有指定一个head（？？？）时，哪个SignatureDef会被服务到。
    export_outputs = {tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: tf.estimator.export.PredictOutput(predictions)}
    # Provide an estimator spec for `ModeKeys.PREDICT`
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=predictions,
                export_outputs=export_outputs)

    #------bulid loss------
    # todo sigmoid_cross_entropy_with_logits(_sentinel=None,labels=None,logits=None,name=None)
    #  该损失函数的输入不需要对网络的输出进行one hot处理，网络输出即是函数logits参数的输入
    #  https://blog.csdn.net/m0_37393514/article/details/81393819       -- 看图片的公式便可知道
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y, labels=labels)) + \
        l2_reg * tf.nn.l2_loss(Feat_Bias) + l2_reg * tf.nn.l2_loss(Feat_Emb)

    # Provide an estimator spec for `ModeKeys.EVAL`
    # todo https://blog.csdn.net/qq_32806793/article/details/85010302
    #  tf.estimator.EstimatorSpec讲解
    #   是一个class，是定义在model_fn中的，并且model_fn返回的也是它的一个实例，这个实例是用来初始化Estimator类的
    #  主要参数：
    #   mode: A ModeKeys. Specifies if this is training, evaluation or prediction.
    #   predictions: Predictions Tensor or dict of Tensor.
    # 	loss: Training loss Tensor. Must be either scalar, or with shape [1].
    # 	train_op: Op for the training step.
    # 	eval_metric_ops: Dict of metric results keyed by name. The values of the dict can be one of the following: (1) instance of Metric class. (2) Results of calling a metric function, namely a (metric_tensor, update_op) tuple. metric_tensor should be evaluated without any impact on state (typically is a pure computation results based on variables.). For example, it should not trigger the update_op or requires any input fetching.
    # 	export_outputs: Describes the output signatures to be exported to SavedModel and used during serving. A dict {name: output} where:
    # 	name: An arbitrary name for this output.
    # 	output: an ExportOutput object such as ClassificationOutput, RegressionOutput, or PredictOutput. Single-headed models only need to specify one entry in this dictionary. Multi-headed models should specify one entry for each head, one of which must be named using signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY. If no entry is provided, a default PredictOutput mapping to predictions will be created.
    # 	training_chief_hooks: Iterable of tf.train.SessionRunHook objects to run on the chief worker during training.
    # 	training_hooks: Iterable of tf.train.SessionRunHook objects to run on all workers during training.
    # 	scaffold: A tf.train.Scaffold object that can be used to set initialization, saver, and more to be used in training.
    # 	evaluation_hooks: Iterable of tf.train.SessionRunHook objects to run during evaluation.
    # 	prediction_hooks: Iterable of tf.train.SessionRunHook objects to run during predictions.
    #  	说明
    # 	根据不同的mode值，使用不同的参数创建不同的EstimatorSpec实例（主要是 训练train，验证dev，测试test）：
    # 	For mode==ModeKeys.TRAIN: 需要的参数是 loss and train_op.
    # 	For mode==ModeKeys.EVAL:  需要的参数是  loss.
    # 	For mode==ModeKeys.PREDICT: 需要的参数是 predictions.
    # 	EstimatorSpec实例定义在方法mode_fn中，方法mode_fn可以计算各个mode下的参数需求，定义好的 EstimatorSpec 用来初始化 一个Estimator实例，
    # 	同时Estimator实例可以根据mode的不同自动的忽视一些参数（操作），例如：train_op will be ignored in eval and infer modes.
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

def set_dist_env():
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
    print('attention_layers ', FLAGS.attention_layers)
    print('dropout ', FLAGS.dropout)
    print('loss_type ', FLAGS.loss_type)
    print('optimizer ', FLAGS.optimizer)
    print('learning_rate ', FLAGS.learning_rate)
    print('l2_reg ', FLAGS.l2_reg)

    #------init Envs------
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
        "attention_layers": FLAGS.attention_layers,
        "dropout": FLAGS.dropout
    }
    config = tf.estimator.RunConfig().replace(session_config = tf.ConfigProto(device_count={'GPU':0, 'CPU':FLAGS.num_threads}),
            log_step_count_steps=FLAGS.log_steps, save_summary_steps=FLAGS.log_steps)
    # todo https://blog.csdn.net/khy19940520/article/details/99948848
    #  replace之后，前面设置的一些参数均失效了
    #  https://www.cnblogs.com/zongfa/p/10149483.html
    #  config 参数为 tf.estimator.RunConfig 对象，包含了执行环境的信息。如果没有传递 config，则它会被 Estimator 实例化，使用的是默认配置。
    Estimator = tf.estimator.Estimator(model_fn=model_fn, model_dir=FLAGS.model_dir, params=model_params, config=config)

    if FLAGS.task_type == 'train':
        train_spec = tf.estimator.TrainSpec(input_fn=lambda: input_fn(tr_files, num_epochs=FLAGS.num_epochs, batch_size=FLAGS.batch_size),max_steps=202)
        eval_spec = tf.estimator.EvalSpec(input_fn=lambda: input_fn(va_files, num_epochs=1, batch_size=FLAGS.batch_size), steps=None, start_delay_secs=1000, throttle_secs=1200)
        tf.estimator.train_and_evaluate(Estimator, train_spec, eval_spec)
    elif FLAGS.task_type == 'eval':
        Estimator.evaluate(input_fn=lambda: input_fn(va_files, num_epochs=1, batch_size=FLAGS.batch_size))
    elif FLAGS.task_type == 'infer':
        preds = Estimator.predict(input_fn=lambda: input_fn(te_files, num_epochs=1, batch_size=FLAGS.batch_size), predict_keys="prob")
        with open(FLAGS.data_dir+"/pred.txt", "w") as fo:
            for prob in preds:
                fo.write("%f\n" % (prob['prob']))
    elif FLAGS.task_type == 'export':
        # todo 开启服务
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
        Estimator.export_savedmodel(FLAGS.servable_model_dir, serving_input_receiver_fn)

if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()


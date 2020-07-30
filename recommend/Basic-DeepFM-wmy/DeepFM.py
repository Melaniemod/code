# -*-coding:utf-8-*-


import numpy as np
import tensorflow as tf
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import roc_auc_score
from time import time
from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm
from yellowfin import YFOptimizer


class DeepFM(BaseEstimator, TransformerMixin):

    def __init__(self, feature_size,
                 field_size,
                 embedding_size,
                 epoch,
                 learning_rate,
                 optimizer_type,
                 batch_norm,
                 batch_norm_decay,
                 dropout_fm,
                 deep_layers,
                 dropout_deep,
                 deep_layers_activation=tf.nn.relu,
                 use_fm=True,
                 use_deep=True,
                 l2_reg=0.0,
                 verbose=1,
                 random_seed=2020,
                 loss_type="logloss",
                 eval_metric=roc_auc_score,
                 is_greater_better=True,
                 batch_size=256):
        assert (use_deep or use_fm)
        assert loss_type in ["mse", "logloss"], "loss_type 只能是 mse, logloss 两者之一"

        self.field_size = field_size
        self.feature_size = feature_size
        self.embedding_size = embedding_size

        self.drop_fm = dropout_fm
        self.deep_layer = deep_layers
        self.deep_layer_activation = deep_layers_activation
        self.drop_deep = dropout_deep
        self.use_fm = use_fm
        self.use_deep = use_deep
        self.l2_reg = l2_reg

        self.epoch = epoch
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.opt_type = optimizer_type

        self.batch_norm = batch_norm
        self.batch_norm_decay = batch_norm_decay

        self.verbose = verbose
        self.rand_seed = random_seed
        self.loss_type = loss_type
        self.eval_emtric = eval_metric
        self.is_greater_better = is_greater_better
        self.train_result, self.valid_result = [], []

        self._init_graph()

    def _init_graph(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            tf.set_random_seed(self.rand_seed)

            self.feat_index = tf.placeholder(tf.int32, shape=[None, None], name="feat_index")
            self.feat_value = tf.placeholder(tf.float32, shape=[None, None], name="feat_value")
            self.label = tf.placeholder(tf.float32, shape=[None, 1], name="label")
            self.dropout_keep_fm = tf.placeholder(tf.float32, shape=[None], name="dropout_keep_fm")
            self.dropout_keep_deep = tf.placeholder(tf.float32, shape=[None], name="dropout_keep_deep")
            self.train_phase = tf.placeholder(tf.bool, name="train_phase")

            self.weight = self._init_weight()

            # 创建DeepFM模型图
            self.embeddings = tf.nn.embedding_lookup(self.weight['feat_embeddings'], self.feat_index)
            feat_value = tf.reshape(self.feat_value, shape=[-1, self.field_size, 1])
            self.embeddings = tf.multiply(self.embeddings, feat_value)  # None*F*K

            # 一阶项
            self.y_first_order = tf.nn.embedding_lookup(self.weight["feat_bias"], self.feat_index)
            self.y_first_order = tf.reduce_sum(tf.multiply(self.y_first_order, feat_value), 2)
            self.y_first_order = tf.nn.dropout(self.y_first_order, self.dropout_keep_fm[0])

            # 二阶项
            # 先求和后平方
            sum_feat_emb = tf.reduce_sum(self.embeddings, 1)
            self.sum_square_feat_emb = tf.square(sum_feat_emb)

            # 先平方后求和
            sqrt_feat_emb = tf.square(self.embeddings)
            self.sqrt_sum_feat_emb = tf.reduce_sum(sqrt_feat_emb, 1)

            # 二阶项
            self.y_second_order = 0.5 * tf.subtract(self.sum_square_feat_emb, self.sqrt_sum_feat_emb)
            self.y_second_order = tf.nn.dropout(self.y_second_order, self.dropout_keep_fm[1])

            # deep部分
            self.y_deep = tf.reshape(self.embeddings, [-1, self.field_size * self.embedding_size])
            self.y_deep = tf.nn.dropout(self.y_deep, self.dropout_keep_deep[0])
            for i in range(0, len(self.deep_layer)):
                self.y_deep = tf.add(tf.matmul(self.y_deep, self.weight["layer_%d" % i]), self.weight["bias_%d" % i])
                if self.batch_norm:
                    self.y_deep = self.batch_norm_layer(self.y_deep, self.train_phase, scope_bn="bn_%d" % i)
                self.y_deep = self.deep_layer_activation(self.y_deep)

            # 三部分向量拼接
            if self.use_deep and self.use_fm:
                concat_input = tf.concat([self.y_first_order, self.y_second_order, self.y_deep],axis =1)
            elif self.use_fm:
                concat_input = tf.concat([self.y_first_order, self.y_second_order],axis=1)
            elif self.use_deep:
                concat_input =self.y_deep
            self.out = tf.add(tf.matmul(concat_input, self.weight['concat_projection']), self.weight['concat_bias'])

            # 损失函数
            if self.loss_type == 'logloss':
                self.out = tf.sigmoid(self.out)
                self.loss = tf.losses.log_loss(self.label, self.out)
            elif self.loss_type == 'mse':
                self.loss = tf.nn.l2_loss(tf.subtract(self.label, self.out))

            # 是否加正则
            if self.l2_reg > 0:
                self.loss += tf.contrib.layers.l2_regularizer(self.l2_reg)(self.weight["concat_projection"])
                if self.use_deep:
                    for i in range(len(self.deep_layer)):
                        self.loss += tf.contrib.layers.l2_regularizer(self.l2_reg)(self.weight["layer_%d" % i])

            # 优化器
            if self.opt_type == 'adam':
                self.opt = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
            elif self.opt_type == 'adagrade':
                self.opt = tf.train.AdagradOptimizer(self.learning_rate).minimize(self.loss)
            elif self.opt_type == 'gd':
                self.opt = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss)
            elif self.opt_type == 'momentum':
                self.opt = tf.train.MomentumOptimizer(self.learning_rate).minimize(self.loss)
            elif self.opt_type == 'yellowfin':
                self.opt = YFOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

            # 建立init
            self.saver = tf.train.Saver()
            init = tf.global_variables_initializer()
            self.sess = self._init_session()
            self.sess.run(init)

            # 打印参数数量
            total_parametres = 0
            for variable in self.weight.values():
                shape = variable.get_shape()
                variable_parametres = 1
                for dim in shape:
                    variable_parametres *= dim
                total_parametres += variable_parametres
            if self.verbose > 0:
                print("# params: %d" % total_parametres)

    def _init_session(self):
        config = tf.ConfigProto(device_count={"gpu": 0})
        config.gpu_options.allow_growth = True
        return tf.Session(config=config)

    def _init_weight(self):
        weights = dict()

        # embedding
        weights["feat_embeddings"] = tf.Variable(
            tf.random_normal([self.feature_size, self.embedding_size], mean=0.0, stddev=0.01),name="feat_embeddings" )
        weights["feat_bias"] = tf.Variable(
            tf.random_uniform([self.feature_size,1], 0.0, 1.0),name='feat_bias')

        # deep
        input_size = self.field_size * self.embedding_size
        # print(f"self.deep_layer[0]={self.deep_layer[0]}")
        # print(f"input_size={input_size}; ")
        # print(f"2.0 / (input_size + self.deep_layer[0])={2.0 / (input_size + self.deep_layer[0])}")
        glorot = np.sqrt(2.0 / (input_size + self.deep_layer[0]))

        # print(f"(input_size,self.deep_layer[0])={(input_size, self.deep_layer[0]), glorot}")
        # print(
        #     f"np.random.normal(loc=0, scale=glorot, size=(input_size, self.deep_layer[0]))={np.random.normal(loc=0, scale=glorot, size=(input_size, self.deep_layer[0]))}")

        weights["layer_0"] = tf.Variable(
            np.random.normal(loc=0, scale=glorot, size=(input_size, self.deep_layer[0])), dtype=np.float32)
        weights["bias_0"] = tf.Variable(np.random.normal(loc=0, scale=glorot, size=(1, self.deep_layer[0])),
                                        dtype=np.float32)  # 1 * layers[0]

        weights["layer_0"] = tf.Variable(
            np.random.normal(loc=0.0, scale=glorot, size=(input_size, self.deep_layer[0])), dtype=tf.float32)
        weights["bias_0"] = tf.Variable(
            np.random.normal(loc=0.0, scale=glorot, size=(1, self.deep_layer[0])), dtype=tf.float32)

        for i in range(1, len(self.deep_layer)):
            glorot = np.sqrt(2 / (self.deep_layer[i - 1] + self.deep_layer[i]))
            weights["layer_%d" % i] = tf.Variable(
                np.random.normal(loc=0.0, scale=glorot, size=[self.deep_layer[i - 1], self.deep_layer[i]]),
                dtype=tf.float32)
            weights["bias_%d" % i] = tf.Variable(np.random.normal(loc=0.0, scale=glorot, size=[1, self.deep_layer[i]]),
                                                 dtype=tf.float32)

        # FM 和 Deep全连接
        if self.use_fm and self.use_deep:
            input_size = self.field_size + self.embedding_size + self.deep_layer[-1]
        elif self.use_fm:
            input_size = self.field_size + self.embedding_size
        elif self.use_deep:
            input_size = self.deep_layer[-1]
        glorot = np.sqrt(2.0 / (input_size + 1))
        weights['concat_projection'] = tf.Variable(np.random.normal(loc=0.0, scale=glorot, size=[input_size, 1]),
                                                   dtype=tf.float32)
        weights['concat_bias'] = tf.Variable(tf.constant(0.01), dtype=tf.float32)
        return weights

    def batch_norm_layer(self, x, train_prase, scope_bn):
        bn_train = batch_norm(x, decay=self.batch_norm_decay, scale=True, scope=scope_bn, is_training=True, reuse=None)
        bn_inference = batch_norm(x, decay=self.batch_norm_decay, scale=True, scope=scope_bn, is_training=False,reuse=True)

        # print(f"type(bn_train)={type(bn_train)}")

        z = tf.cond(train_prase,lambda :bn_train,lambda :bn_inference)
        return z

    def shuffer_in_union_scary(self, a, b, c):
        rng_state = np.random.get_state()
        np.random.shuffle(a)
        np.random.set_state(rng_state)
        np.random.shuffle(b)
        np.random.set_state(rng_state)
        np.random.shuffle(c)

    def fit_on_batch(self, Xi, Xv, y):
        feed_dict = {self.feat_index: Xi
            , self.feat_value: Xv
            , self.label: y
            , self.dropout_keep_fm: self.drop_fm
            , self.dropout_keep_deep: self.drop_deep
            , self.train_phase: True}

        loss, opt = self.sess.run((self.loss, self.opt), feed_dict=feed_dict)
        return loss

    def get_batch(self, Xi, Xv, y, batch_size, index):
        start = batch_size * index
        end = batch_size * (index + 1)
        end = end if end < len(Xv) else len(Xv)
        return Xi[start:end], Xv[start:end], [[y_] for y_ in y[start:end]]

    def training_termination(self, valid_result):
        if len(valid_result) > 5:
            if self.is_greater_better:
                if valid_result[-1] < valid_result[-2] \
                        and valid_result[-2] < valid_result[-3] \
                        and valid_result[-3] < valid_result[-4] \
                        and valid_result[-4] < valid_result[-5]:
                    return True
            else:
                if valid_result[-1] > valid_result[-2] \
                        and valid_result[-2] > valid_result[-3] \
                        and valid_result[-3] > valid_result[-4] \
                        and valid_result[-4] > valid_result[-5]:
                    return True
        return False

    def evaluate(self, Xi, Xv, y):
        y_pred = self.predict(Xi, Xv)
        return self.eval_emtric(y, y_pred)

    def fit(self, Xi_train, Xv_train, y_train, Xi_valid, Xv_valid, y_valid, early_stopping=False, refit=False):

        # print("begin fit ... ")
        has_valid = Xi_valid is not None
        for epoch in range(self.epoch):
            self.shuffer_in_union_scary(Xi_train, Xv_train, y_train)
            total_batch = int(len(Xi_train) / self.batch_size)
            # print(f"total_batch={total_batch}")

            for i in range(total_batch):
                Xi_batch, Xv_batch, y_batch = self.get_batch(Xi_train, Xv_train, y_train, self.batch_size, i)
                # print(f"i={i}")
                self.fit_on_batch(Xi_batch, Xv_batch, y_batch)

            train_result = self.evaluate(Xi_batch, Xv_batch, y_batch)
            self.train_result.append(train_result)

            if has_valid:
                valid_result = self.evaluate(Xi_valid, Xv_valid, y_valid)
                self.valid_result.append(valid_result)

            # print(f"verbose={self.verbose}")
            if self.verbose > 0 and epoch % self.verbose == 0:
                if has_valid:
                    print(f"[{epoch + 1}] train-result={train_result}; valid-result={valid_result}")
                else:
                    print(f"[{epoch + 1}] train-result={train_result}")

            if early_stopping and self.training_termination(self.valid_result):
                break

        if refit and has_valid:
            if self.is_greater_better:
                best_result = max(self.valid_result)
            else:
                best_result = min(self.train_result)
            Xi_train = Xi_train + Xi_valid
            Xv_train = Xv_train + Xv_train
            y_train = y_train + y_valid

            for epoch in range(100):
                self.shuffer_in_union_scary(Xi_train, Xv_train, y_train)
                batch_total = int(len(Xi_train) / self.batch_size)

                for i in batch_total:
                    Xi_batch, Xv_batch, y_batch = self.get_batch(Xi_train, Xv_train, y_valid, self.batch_size, i)
                    self.fit_on_batch(Xi_batch, Xv_batch, y_batch)

                train_result = self.evaluate(Xi_train, Xv_train, y_train)
                if abs(train_result - best_result) < 0.001 or \
                        (self.is_greater_better and train_result > best_result) or \
                        (self.is_greater_better and train_result < best_result):
                    break

    def predict(self, Xi, Xv):
        dummy_y = [1] * len(Xi)
        batch_index = 0
        Xi_batch, Xv_batch, y_batch = self.get_batch(Xi, Xv, dummy_y, self.batch_size, batch_index)
        # print(f"predict,{len(Xi),len(Xi_batch)}")

        while len(Xi_batch) > 0:
            num_batch = len(Xi_batch)
            feed_dict = {self.feat_index: Xi_batch
                , self.feat_value: Xv_batch
                , self.label: y_batch
                , self.dropout_keep_fm: [1.0] * len(self.drop_fm)
                , self.dropout_keep_deep: [1.0] * len(self.drop_deep)
                , self.train_phase: False
                         }
            batch_out = self.sess.run(self.out, feed_dict=feed_dict)

            if batch_index == 0:
                y_pred = np.reshape(batch_out, (num_batch, ))
            else:
                y_pred = np.concatenate((y_pred, np.reshape(batch_out, (num_batch,))))
            batch_index += 1
            Xi_batch, Xv_batch, y_batch = self.get_batch(Xi, Xv, dummy_y, self.batch_size, batch_index)
        return y_pred



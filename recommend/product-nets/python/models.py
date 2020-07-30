# from __future__ import print_function
# from __future__ import division
# from __future__ import absolute_import

import sys
if sys.version[0] == '2':
    import cPickle as pkl
else:
    import pickle as pkl

import numpy as np
import tensorflow as tf

from python import utils

dtype = utils.DTYPE


class Model:
    def __init__(self):
        self.sess = None
        self.X = None
        self.y = None
        self.layer_keeps = None
        self.vars = None
        self.keep_prob_train = None
        self.keep_prob_test = None

    def run(self, fetches, X=None, y=None, mode='train'):
            feed_dict = {}
            # todo type(self.X)=<class 'list'>；这里 self.X 是一个list是因为PNN 中 "self.X = [tf.sparse_placeholder(dtype) for i in range(num_inputs)]" 定义的是list
            #  可以任务 PNN 类中先指定了 self.X(list), 这里对PNN中定义的self.X 赋值了
            # print(f"type(self.X)={type(self.X)}")
            if type(self.X) is list:
                for i in range(len(X)):
                    # todo 这里每个 X[i] 都是一个对象，下面这样明明只有一个输入数据却分成了16组喂入，后面取数据的时候是分为每个 field 取的吗
                    feed_dict[self.X[i]] = X[i]
            else:
                feed_dict[self.X] = X
            #todo  feed_dict=16
            # print(f"feed_dict={len(feed_dict)}")
            if y is not None:
                feed_dict[self.y] = y
            if self.layer_keeps is not None:
                if mode == 'train':
                    # todo 这里 self.layer_keeps 是一个容器，self.keep_prob_train是一个数值；根据模型的不同阶段，看喂入 self.keep_prob_test 还是喂入 self.keep_prob_train
                    feed_dict[self.layer_keeps] = self.keep_prob_train
                elif mode == 'test':
                    feed_dict[self.layer_keeps] = self.keep_prob_test
            return self.sess.run(fetches, feed_dict)

    def dump(self, model_path):
        var_map = {}
        for name, var in self.vars.iteritems():
            var_map[name] = self.run(var)
        pkl.dump(var_map, open(model_path, 'wb'))
        print('model dumped at', model_path)


class LR(Model):
    def __init__(self, input_dim=None, output_dim=1, init_path=None, opt_algo='gd', learning_rate=1e-2, l2_weight=0,
                 random_seed=None):
        Model.__init__(self)
        init_vars = [('w', [input_dim, output_dim], 'xavier', dtype),
                     ('b', [output_dim], 'zero', dtype)]
        self.graph = tf.Graph()
        with self.graph.as_default():
            if random_seed is not None:
                tf.set_random_seed(random_seed)
            self.X = tf.sparse_placeholder(dtype)
            self.y = tf.placeholder(dtype)
            self.vars = utils.init_var_map(init_vars, init_path)

            w = self.vars['w']
            b = self.vars['b']
            xw = tf.sparse_tensor_dense_matmul(self.X, w)
            logits = tf.reshape(xw + b, [-1])
            self.y_prob = tf.sigmoid(logits)

            self.loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(labels=self.y, logits=logits)) + \
                        l2_weight * tf.nn.l2_loss(xw)
            self.optimizer = utils.get_optimizer(opt_algo, learning_rate, self.loss)

            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=config)
            tf.global_variables_initializer().run(session=self.sess)


class FM(Model):
    def __init__(self, input_dim=None, output_dim=1, factor_order=10, init_path=None, opt_algo='gd', learning_rate=1e-2,
                 l2_w=0, l2_v=0, random_seed=None):
        Model.__init__(self)
        init_vars = [('w', [input_dim, output_dim], 'xavier', dtype),
                     ('v', [input_dim, factor_order], 'xavier', dtype),
                     ('b', [output_dim], 'zero', dtype)]
        self.graph = tf.Graph()
        with self.graph.as_default():
            if random_seed is not None:
                tf.set_random_seed(random_seed)
            self.X = tf.sparse_placeholder(dtype)
            self.y = tf.placeholder(dtype)
            self.vars = utils.init_var_map(init_vars, init_path)

            w = self.vars['w']
            v = self.vars['v']
            b = self.vars['b']

            X_square = tf.SparseTensor(self.X.indices, tf.square(self.X.values), tf.to_int64(tf.shape(self.X)))
            xv = tf.square(tf.sparse_tensor_dense_matmul(self.X, v))
            p = 0.5 * tf.reshape(
                tf.reduce_sum(xv - tf.sparse_tensor_dense_matmul(X_square, tf.square(v)), 1),
                [-1, output_dim])
            xw = tf.sparse_tensor_dense_matmul(self.X, w)
            logits = tf.reshape(xw + b + p, [-1])
            self.y_prob = tf.sigmoid(logits)

            self.loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=self.y)) + \
                        l2_w * tf.nn.l2_loss(xw) + \
                        l2_v * tf.nn.l2_loss(xv)
            self.optimizer = utils.get_optimizer(opt_algo, learning_rate, self.loss)

            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=config)
            tf.global_variables_initializer().run(session=self.sess)


class FNN(Model):
    def __init__(self, field_sizes=None, embed_size=10, layer_sizes=None, layer_acts=None, drop_out=None,
                 embed_l2=None, layer_l2=None, init_path=None, opt_algo='gd', learning_rate=1e-2, random_seed=None):
        Model.__init__(self)
        init_vars = []
        num_inputs = len(field_sizes)
        for i in range(num_inputs):
            init_vars.append(('embed_%d' % i, [field_sizes[i], embed_size], 'xavier', dtype))
        node_in = num_inputs * embed_size
        for i in range(len(layer_sizes)):
            init_vars.append(('w%d' % i, [node_in, layer_sizes[i]], 'xavier', dtype))
            init_vars.append(('b%d' % i, [layer_sizes[i]], 'zero', dtype))
            node_in = layer_sizes[i]
        self.graph = tf.Graph()
        with self.graph.as_default():
            if random_seed is not None:
                tf.set_random_seed(random_seed)
            self.X = [tf.sparse_placeholder(dtype) for i in range(num_inputs)]
            self.y = tf.placeholder(dtype)
            self.keep_prob_train = 1 - np.array(drop_out)
            self.keep_prob_test = np.ones_like(drop_out)
            self.layer_keeps = tf.placeholder(dtype)
            self.vars = utils.init_var_map(init_vars, init_path)
            w0 = [self.vars['embed_%d' % i] for i in range(num_inputs)]
            xw = tf.concat([tf.sparse_tensor_dense_matmul(self.X[i], w0[i]) for i in range(num_inputs)], 1)
            l = xw

            for i in range(len(layer_sizes)):
                wi = self.vars['w%d' % i]
                bi = self.vars['b%d' % i]
                print(l.shape, wi.shape, bi.shape)
                l = tf.nn.dropout(
                    utils.activate(
                        tf.matmul(l, wi) + bi,
                        layer_acts[i]),
                    self.layer_keeps[i])

            l = tf.squeeze(l)
            self.y_prob = tf.sigmoid(l)

            self.loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=l, labels=self.y))
            if layer_l2 is not None:
                self.loss += embed_l2 * tf.nn.l2_loss(xw)
                for i in range(len(layer_sizes)):
                    wi = self.vars['w%d' % i]
                    self.loss += layer_l2[i] * tf.nn.l2_loss(wi)
            self.optimizer = utils.get_optimizer(opt_algo, learning_rate, self.loss)

            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=config)
            tf.global_variables_initializer().run(session=self.sess)


class DeepFM(Model):
    def __init__(self, field_sizes=None, embed_size=10, layer_sizes=None, layer_acts=None, drop_out=None,
                 embed_l2=None, layer_l2=None, init_path=None, opt_algo='gd', learning_rate=1e-2, random_seed=None):
        Model.__init__(self)
        init_vars = []
        num_inputs = len(field_sizes)
        for i in range(num_inputs):
            init_vars.append(('embed_%d' % i, [field_sizes[i], embed_size], 'xavier', dtype))
            init_vars.append(('weight_%d' % i, [field_sizes[i], 1], 'xavier', dtype))
            init_vars.append(('bias', [1], 'zero', dtype))
        node_in = num_inputs * embed_size
        for i in range(len(layer_sizes)):
            init_vars.append(('w%d' % i, [node_in, layer_sizes[i]], 'xavier', dtype))
            init_vars.append(('b%d' % i, [layer_sizes[i]], 'zero', dtype))
            node_in = layer_sizes[i]
        self.graph = tf.Graph()
        with self.graph.as_default():
            if random_seed is not None:
                tf.set_random_seed(random_seed)
            self.X = [tf.sparse_placeholder(dtype) for i in range(num_inputs)]
            self.y = tf.placeholder(dtype)
            self.keep_prob_train = 1 - np.array(drop_out)
            self.keep_prob_test = np.ones_like(drop_out)
            self.layer_keeps = tf.placeholder(dtype)
            self.vars = utils.init_var_map(init_vars, init_path)
            w = [self.vars['weight_%d' % i] for i in range(num_inputs)]
            v = [self.vars['embed_%d' % i] for i in range(num_inputs)]
            b = self.vars['bias']
            xw = tf.concat([tf.sparse_tensor_dense_matmul(self.X[i], w[i]) for i in range(num_inputs)], 1)
            xv = tf.concat([tf.sparse_tensor_dense_matmul(self.X[i], v[i]) for i in range(num_inputs)], 1)
            l = xv

            for i in range(len(layer_sizes)):
                wi = self.vars['w%d' % i]
                bi = self.vars['b%d' % i]
                print(l.shape, wi.shape, bi.shape)
                l = tf.nn.dropout(
                    utils.activate(
                        tf.matmul(l, wi) + bi,
                        layer_acts[i]),
                    self.layer_keeps[i])
            l = tf.squeeze(l)

            xv = tf.reshape(xv, [-1, num_inputs, embed_size])
            p = 0.5 * tf.reduce_sum(
                tf.square(tf.reduce_sum(xv, 1)) -
                tf.reduce_sum(tf.square(xv), 1),
            1)
            xw = tf.reduce_sum(xw, 1)
            logits = tf.reshape(l + xw + b + p, [-1])

            self.y_prob = tf.sigmoid(logits)

            self.loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=self.y))
            if layer_l2 is not None:
                self.loss += embed_l2 * tf.nn.l2_loss(xw)
                for i in range(len(layer_sizes)):
                    wi = self.vars['w%d' % i]
                    self.loss += layer_l2[i] * tf.nn.l2_loss(wi)
            self.optimizer = utils.get_optimizer(opt_algo, learning_rate, self.loss)

            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=config)
            tf.global_variables_initializer().run(session=self.sess)


class CCPM(Model):
    def __init__(self, field_sizes=None, embed_size=10, filter_sizes=None, layer_acts=None, drop_out=None,
                 init_path=None, opt_algo='gd', learning_rate=1e-2, random_seed=None):
        Model.__init__(self)
        init_vars = []
        num_inputs = len(field_sizes)
        for i in range(num_inputs):
            init_vars.append(('embed_%d' % i, [field_sizes[i], embed_size], 'xavier', dtype))
        init_vars.append(('f1', [embed_size, filter_sizes[0], 1, 2], 'xavier', dtype))
        init_vars.append(('f2', [embed_size, filter_sizes[1], 2, 2], 'xavier', dtype))
        init_vars.append(('w1', [2 * 3 * embed_size, 1], 'xavier', dtype))
        init_vars.append(('b1', [1], 'zero', dtype))

        self.graph = tf.Graph()
        with self.graph.as_default():
            if random_seed is not None:
                tf.set_random_seed(random_seed)
            self.X = [tf.sparse_placeholder(dtype) for i in range(num_inputs)]
            self.y = tf.placeholder(dtype)
            self.keep_prob_train = 1 - np.array(drop_out)
            self.keep_prob_test = np.ones_like(drop_out)
            self.layer_keeps = tf.placeholder(dtype)
            self.vars = utils.init_var_map(init_vars, init_path)
            w0 = [self.vars['embed_%d' % i] for i in range(num_inputs)]
            xw = tf.concat([tf.sparse_tensor_dense_matmul(self.X[i], w0[i]) for i in range(num_inputs)], 1)
            l = xw

            l = tf.transpose(tf.reshape(l, [-1, num_inputs, embed_size, 1]), [0, 2, 1, 3])
            f1 = self.vars['f1']
            l = tf.nn.conv2d(l, f1, [1, 1, 1, 1], 'SAME')
            l = tf.transpose(
                utils.max_pool_4d(
                    tf.transpose(l, [0, 1, 3, 2]),
                    int(num_inputs / 2)),
                [0, 1, 3, 2])
            f2 = self.vars['f2']
            l = tf.nn.conv2d(l, f2, [1, 1, 1, 1], 'SAME')
            l = tf.transpose(
                utils.max_pool_4d(
                    tf.transpose(l, [0, 1, 3, 2]), 3),
                [0, 1, 3, 2])
            l = tf.nn.dropout(
                utils.activate(
                    tf.reshape(l, [-1, embed_size * 3 * 2]),
                    layer_acts[0]),
                self.layer_keeps[0])
            w1 = self.vars['w1']
            b1 = self.vars['b1']
            l = tf.matmul(l, w1) + b1

            l = tf.squeeze(l)
            self.y_prob = tf.sigmoid(l)

            self.loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=l, labels=self.y))
            self.optimizer = utils.get_optimizer(opt_algo, learning_rate, self.loss)

            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=config)
            tf.global_variables_initializer().run(session=self.sess)


class PNN1(Model):
    def __init__(self, field_sizes=None, embed_size=10, layer_sizes=None, layer_acts=None, drop_out=None,
                 embed_l2=None, layer_l2=None, init_path=None, opt_algo='gd', learning_rate=1e-2, random_seed=None):
        # todo {'field_sizes': [25, 131235, 35, 367, 2, 800, 961, 2, 2, 4, 2, 4, 2, 14, 5, 5],
        #  'embed_size': 10, 'layer_sizes': [500, 1], 'layer_acts': ['relu', None],
        #  'drop_out': [0, 0], 'opt_algo': 'gd', 'learning_rate': 0.1,
        #  'embed_l2': 0, 'layer_l2': [0.0, 0.0], 'random_seed': 0,
        #  'layer_norm': True}
        Model.__init__(self)
        init_vars = []
        num_inputs = len(field_sizes)
        # todo num_inputs=16
        print(f"num_inputs={num_inputs}")
        for i in range(num_inputs):
            init_vars.append(('embed_%d' % i, [field_sizes[i], embed_size], 'xavier', dtype))

        num_pairs = int(num_inputs * (num_inputs - 1) / 2)
        # todo 该节点输入的维度；开始输入deep的维度就是:线性部分的维度(num_inputs * embed_size)+线性部分的维度(num_pairs)
        #  看了几个 code 都是这样直接使用 embedding 做交叉，而不是用 embedding 再做一个权重矩阵。这样 PNN 的非线性部分就是FM对吧？
        #  这里非线性项最后输出的大小是 num_pairs，交叉之后做了加和
        node_in = num_inputs * embed_size + num_pairs
        # node_in = num_inputs * (embed_size + num_inputs)
        # todo 这里输出层只有一个神经元，如何判断该推荐哪个商品呢？
        for i in range(len(layer_sizes)):
            # todo xavier 一种初始化方式，在note -- 外链--疑问中有说明
            init_vars.append(('w%d' % i, [node_in, layer_sizes[i]], 'xavier', dtype))
            init_vars.append(('b%d' % i, [layer_sizes[i]], 'zero', dtype))
            node_in = layer_sizes[i]
        # todo init_vars=[('embed_0', [25, 10], 'xavier', tf.float32),
        #  ('embed_1', [131235, 10], 'xavier', tf.float32),
        #  ('embed_2', [35, 10], 'xavier', tf.float32), ('embed_3', [367, 10], 'xavier', tf.float32),
        #  ('embed_4', [2, 10], 'xavier', tf.float32), ('embed_5', [800, 10], 'xavier', tf.float32),
        #  ('embed_6', [961, 10], 'xavier', tf.float32), ('embed_7', [2, 10], 'xavier', tf.float32),
        #  ('embed_8', [2, 10], 'xavier', tf.float32), ('embed_9', [4, 10], 'xavier', tf.float32),
        #  ('embed_10', [2, 10], 'xavier', tf.float32), ('embed_11', [4, 10], 'xavier', tf.float32),
        #  ('embed_12', [2, 10], 'xavier', tf.float32), ('embed_13', [14, 10], 'xavier', tf.float32),
        #  ('embed_14', [5, 10], 'xavier', tf.float32), ('embed_15', [5, 10], 'xavier', tf.float32),
        #  ('w0', [280, 500], 'xavier', tf.float32),
        #  ('b0', [500], 'zero', tf.float32), ('w1', [500, 1], 'xavier', tf.float32),
        #  ('b1', [1], 'zero', tf.float32)]
        # print(f"init_vars={init_vars}")

        self.graph = tf.Graph()
        with self.graph.as_default():
            if random_seed is not None:
                tf.set_random_seed(random_seed)
            # todo 这里 X 的 placeholder 是一个有16个元素的list
            self.X = [tf.sparse_placeholder(dtype) for i in range(num_inputs)]
            self.y = tf.placeholder(dtype)
            self.keep_prob_train = 1 - np.array(drop_out)
            self.keep_prob_test = np.ones_like(drop_out)
            self.layer_keeps = tf.placeholder(dtype)

            self.vars = utils.init_var_map(init_vars, init_path)

            # todo 这里是分 field 创建 emedding 权重矩阵的，而不是所有field一起创建一个embedding矩阵，然后lookup查找权重
            w0 = [self.vars['embed_%d' % i] for i in range(num_inputs)]
            # todo sparse_tensor_dense_matmul(sp_a,b,adjoint_a=False,adjoint_b=False,name=None)
            #  用稠密矩阵“b”乘以 SparseTensor(秩为 2)“sp_a”.
            #  这里为什么不用 embedding_lookup 呢？
            #  这里
            xw = tf.concat([tf.sparse_tensor_dense_matmul(self.X[i], w0[i]) for i in range(num_inputs)], 1)
            xw3d = tf.reshape(xw, [-1, num_inputs, embed_size])
            # todo self.vars={'embed_0': <tf.Variable 'embed_0:0' shape=(25, 10) dtype=float32_ref>,
            #  'embed_1': <tf.Variable 'embed_1:0' shape=(131235, 10) dtype=float32_ref>,
            #  'embed_2': <tf.Variable 'embed_2:0' shape=(35, 10) dtype=float32_ref>,
            #  'embed_3': <tf.Variable 'embed_3:0' shape=(367, 10) dtype=float32_ref>,
            #  'embed_4': <tf.Variable 'embed_4:0' shape=(2, 10) dtype=float32_ref>,
            #  'embed_5': <tf.Variable 'embed_5:0' shape=(800, 10) dtype=float32_ref>,
            #  'embed_6': <tf.Variable 'embed_6:0' shape=(961, 10) dtype=float32_ref>,
            #  'embed_7': <tf.Variable 'embed_7:0' shape=(2, 10) dtype=float32_ref>,
            #  'embed_8': <tf.Variable 'embed_8:0' shape=(2, 10) dtype=float32_ref>,
            #  'embed_9': <tf.Variable 'embed_9:0' shape=(4, 10) dtype=float32_ref>,
            #  'embed_10': <tf.Variable 'embed_10:0' shape=(2, 10) dtype=float32_ref>,
            #  'embed_11': <tf.Variable 'embed_11:0' shape=(4, 10) dtype=float32_ref>,
            #  'embed_12': <tf.Variable 'embed_12:0' shape=(2, 10) dtype=float32_ref>,
            #  'embed_13': <tf.Variable 'embed_13:0' shape=(14, 10) dtype=float32_ref>,
            #  'embed_14': <tf.Variable 'embed_14:0' shape=(5, 10) dtype=float32_ref>,
            #  'embed_15': <tf.Variable 'embed_15:0' shape=(5, 10) dtype=float32_ref>,
            #  'w0': <tf.Variable 'w0:0' shape=(280, 500) dtype=float32_ref>,
            #  'b0': <tf.Variable 'b0:0' shape=(500,) dtype=float32_ref>,
            #  'w1': <tf.Variable 'w1:0' shape=(500, 1) dtype=float32_ref>,
            #  'b1': <tf.Variable 'b1:0' shape=(1,) dtype=float32_ref>}
            # print(f"self.vars={self.vars}")

            # todo w0=[<tf.Variable 'embed_0:0' shape=(25, 10) dtype=float32_ref>,
            #  <tf.Variable 'embed_1:0' shape=(131235, 10) dtype=float32_ref>,
            #  <tf.Variable 'embed_2:0' shape=(35, 10) dtype=float32_ref>,
            #  <tf.Variable 'embed_3:0' shape=(367, 10) dtype=float32_ref>,
            #  <tf.Variable 'embed_4:0' shape=(2, 10) dtype=float32_ref>,
            #  <tf.Variable 'embed_5:0' shape=(800, 10) dtype=float32_ref>,
            #  <tf.Variable 'embed_6:0' shape=(961, 10) dtype=float32_ref>,
            #  <tf.Variable 'embed_7:0' shape=(2, 10) dtype=float32_ref>,
            #  <tf.Variable 'embed_8:0' shape=(2, 10) dtype=float32_ref>,
            #  <tf.Variable 'embed_9:0' shape=(4, 10) dtype=float32_ref>,
            #  <tf.Variable 'embed_10:0' shape=(2, 10) dtype=float32_ref>,
            #  <tf.Variable 'embed_11:0' shape=(4, 10) dtype=float32_ref>,
            #  <tf.Variable 'embed_12:0' shape=(2, 10) dtype=float32_ref>,
            #  <tf.Variable 'embed_13:0' shape=(14, 10) dtype=float32_ref>,
            #  <tf.Variable 'embed_14:0' shape=(5, 10) dtype=float32_ref>,
            #  <tf.Variable 'embed_15:0' shape=(5, 10) dtype=float32_ref>];
            #  xw=Tensor("concat:0", shape=(?, 160), dtype=float32);
            #  xw3d=Tensor("Reshape:0", shape=(?, 16, 10), dtype=float32)
            # print(f"w0={w0};xw={xw};xw3d={xw3d}")

            row = []
            col = []
            for i in range(num_inputs-1):
                for j in range(i+1, num_inputs):
                    row.append(i)
                    col.append(j)
            # batch * pair * k
            p = tf.transpose(
                # pair * batch * k
                tf.gather(
                    # num * batch * k -- 这里num是 pair 的数量吗？
                    tf.transpose(
                        xw3d, [1, 0, 2]),
                    row),
                [1, 0, 2])
            # batch * pair * k
            q = tf.transpose(
                tf.gather(
                    tf.transpose(
                        xw3d, [1, 0, 2]),
                    col),
                [1, 0, 2])
            # todo p=Tensor("transpose_1:0", shape=(?, 120, 10), dtype=float32)
            #  这里 p,q 都已经是[-1, num_pairs, embed_size]的了吧，为什么还要reshape成这样的呢
            # print(f"p={p}")
            p = tf.reshape(p, [-1, num_pairs, embed_size])
            q = tf.reshape(q, [-1, num_pairs, embed_size])
            # todo 这里不是只有一个 embedding 吗？原论文中 首先有个 W 将原稀疏特征转化为 embedding，然后根据 embedding做线性部分 和非线性部分
            #  这里的交叉是两两 embedding 的内积
            ip = tf.reshape(tf.reduce_sum(p * q, [-1]), [-1, num_pairs])
            # todo 120=15*16
            #  ip=Tensor("Reshape_3:0", shape=(?, 120), dtype=float32);
            #  p=Tensor("Reshape_1:0", shape=(?, 120, 10), dtype=float32);
            #  q=Tensor("Reshape_2:0", shape=(?, 120, 10), dtype=float32)
            # print(f"ip={ip}; p={p}; q={q}")
            # simple but redundant
            # batch * n * 1 * k, batch * 1 * n * k
            # ip = tf.reshape(
            #     tf.reduce_sum(
            #         tf.expand_dims(xw3d, 2) *
            #         tf.expand_dims(xw3d, 1),
            #         3),
            #     [-1, num_inputs**2])
            l = tf.concat([xw, ip], 1)

            for i in range(len(layer_sizes)):
                wi = self.vars['w%d' % i]
                bi = self.vars['b%d' % i]
                l = tf.nn.dropout(
                    utils.activate(
                        tf.matmul(l, wi) + bi,
                        layer_acts[i]),
                    self.layer_keeps[i])
            # todo l=Tensor("dropout_1/mul_1:0", shape=(?, 1), dtype=float32)
            # print(f"l={l}")
            l = tf.squeeze(l)
            self.y_prob = tf.sigmoid(l)
            # todo self.y_prob=Tensor("Sigmoid:0", dtype=float32)
            print(f"self.y_prob={self.y_prob};")

            self.loss = tf.reduce_mean(
                #
                tf.nn.sigmoid_cross_entropy_with_logits(logits=l, labels=self.y))
            if layer_l2 is not None:
                # todo 这里加正则， embedding的权重 和 deep权重都加了，这里 embed_l2=0，相当于没用正则吗？
                #  这里用的是wx--这里就是git中所说的稀疏正则对吧？
                self.loss += embed_l2 * tf.nn.l2_loss(xw)
                for i in range(len(layer_sizes)):
                    wi = self.vars['w%d' % i]
                    self.loss += layer_l2[i] * tf.nn.l2_loss(wi)
            self.optimizer = utils.get_optimizer(opt_algo, learning_rate, self.loss)

            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=config)
            tf.global_variables_initializer().run(session=self.sess)


class PNN2(Model):
    def __init__(self, field_sizes=None, embed_size=10, layer_sizes=None, layer_acts=None, drop_out=None,
                 embed_l2=None, layer_l2=None, init_path=None, opt_algo='gd', learning_rate=1e-2, random_seed=None,
                 layer_norm=True, kernel_type='mat'):
        Model.__init__(self)
        init_vars = []
        num_inputs = len(field_sizes)
        for i in range(num_inputs):
            init_vars.append(('embed_%d' % i, [field_sizes[i], embed_size], 'xavier', dtype))
        num_pairs = int(num_inputs * (num_inputs - 1) / 2)
        node_in = num_inputs * embed_size + num_pairs
        if kernel_type == 'mat':
            init_vars.append(('kernel', [embed_size, num_pairs, embed_size], 'xavier', dtype))
        elif kernel_type == 'vec':
            init_vars.append(('kernel', [num_pairs, embed_size], 'xavier', dtype))
        elif kernel_type == 'num':
            init_vars.append(('kernel', [num_pairs, 1], 'xavier', dtype))
        for i in range(len(layer_sizes)):
            init_vars.append(('w%d' % i, [node_in, layer_sizes[i]], 'xavier', dtype))
            init_vars.append(('b%d' % i, [layer_sizes[i]], 'zero',  dtype))
            node_in = layer_sizes[i]
        self.graph = tf.Graph()
        with self.graph.as_default():
            if random_seed is not None:
                tf.set_random_seed(random_seed)
            self.X = [tf.sparse_placeholder(dtype) for i in range(num_inputs)]
            self.y = tf.placeholder(dtype)
            self.keep_prob_train = 1 - np.array(drop_out)
            self.keep_prob_test = np.ones_like(drop_out)
            self.layer_keeps = tf.placeholder(dtype)
            self.vars = utils.init_var_map(init_vars, init_path)
            w0 = [self.vars['embed_%d' % i] for i in range(num_inputs)]
            xw = tf.concat([tf.sparse_tensor_dense_matmul(self.X[i], w0[i]) for i in range(num_inputs)], 1)
            xw3d = tf.reshape(xw, [-1, num_inputs, embed_size])

            row = []
            col = []
            for i in range(num_inputs - 1):
                for j in range(i + 1, num_inputs):
                    row.append(i)
                    col.append(j)
            # batch * pair * k
            p = tf.transpose(
                # pair * batch * k
                tf.gather(
                    # num * batch * k
                    tf.transpose(
                        xw3d, [1, 0, 2]),
                    row),
                [1, 0, 2])
            # batch * pair * k
            q = tf.transpose(
                tf.gather(
                    tf.transpose(
                        xw3d, [1, 0, 2]),
                    col),
                [1, 0, 2])
            # b * p * k
            p = tf.reshape(p, [-1, num_pairs, embed_size])
            # b * p * k
            q = tf.reshape(q, [-1, num_pairs, embed_size])
            k = self.vars['kernel']

            if kernel_type == 'mat':
                # batch * 1 * pair * k
                p = tf.expand_dims(p, 1)
                # batch * pair
                kp = tf.reduce_sum(
                    # batch * pair * k
                    tf.multiply(
                        # batch * pair * k
                        tf.transpose(
                            # batch * k * pair
                            tf.reduce_sum(
                                # batch * k * pair * k
                                tf.multiply(
                                    p, k),
                                -1),
                            [0, 2, 1]),
                        q),
                    -1)
            else:
                # 1 * pair * (k or 1)
                k = tf.expand_dims(k, 0)
                # batch * pair
                kp = tf.reduce_sum(p * q * k, -1)

            #
            # if layer_norm:
            #     # x_mean, x_var = tf.nn.moments(xw, [1], keep_dims=True)
            #     # xw = (xw - x_mean) / tf.sqrt(x_var)
            #     # x_g = tf.Variable(tf.ones([num_inputs * embed_size]), name='x_g')
            #     # x_b = tf.Variable(tf.zeros([num_inputs * embed_size]), name='x_b')
            #     # x_g = tf.Print(x_g, [x_g[:10], x_b])
            #     # xw = xw * x_g + x_b
            #     p_mean, p_var = tf.nn.moments(op, [1], keep_dims=True)
            #     op = (op - p_mean) / tf.sqrt(p_var)
            #     p_g = tf.Variable(tf.ones([embed_size**2]), name='p_g')
            #     p_b = tf.Variable(tf.zeros([embed_size**2]), name='p_b')
            #     # p_g = tf.Print(p_g, [p_g[:10], p_b])
            #     op = op * p_g + p_b

            l = tf.concat([xw, kp], 1)
            for i in range(len(layer_sizes)):
                wi = self.vars['w%d' % i]
                bi = self.vars['b%d' % i]
                l = tf.nn.dropout(
                    utils.activate(
                        tf.matmul(l, wi) + bi,
                        layer_acts[i]),
                    self.layer_keeps[i])

            l = tf.squeeze(l)
            self.y_prob = tf.sigmoid(l)

            self.loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=l, labels=self.y))
            if layer_l2 is not None:
                self.loss += embed_l2 * tf.nn.l2_loss(xw)#tf.concat(w0, 0))
                for i in range(len(layer_sizes)):
                    wi = self.vars['w%d' % i]
                    self.loss += layer_l2[i] * tf.nn.l2_loss(wi)
            self.optimizer = utils.get_optimizer(opt_algo, learning_rate, self.loss)

            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=config)
            tf.global_variables_initializer().run(session=self.sess)

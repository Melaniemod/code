from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import sys
if sys.version[0] == '2':
    import cPickle as pkl
else:
    import pickle as pkl

import numpy as np
import tensorflow as tf
from scipy.sparse import coo_matrix

DTYPE = tf.float32

FIELD_SIZES = [0] * 26
with open('../data/featindex.txt') as fin:
    for line in fin:
        line = line.strip().split(':')
        if len(line) > 1:
            # todo 为什么这里要 - 1？FIELD_SIZES的顺序重要吗？
            f = int(line[0]) - 1
            FIELD_SIZES[f] += 1
        # print(f"line={line}; f = {f}; ")
        # break
# todo field sizes: [25, 131235, 35, 367, 2, 800, 961, 2, 2, 4, 2, 4, 2, 14, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5]
print('field sizes:', FIELD_SIZES)
FIELD_OFFSETS = [sum(FIELD_SIZES[:i]) for i in range(len(FIELD_SIZES))]
INPUT_DIM = sum(FIELD_SIZES)
OUTPUT_DIM = 1
STDDEV = 1e-3
MINVAL = -1e-3
MAXVAL = 1e-3


def read_data(file_name):
    X = []
    D = []
    y = []
    with open(file_name) as fin:
        for line in fin:
            fields = line.strip().split()
            y_i = int(fields[0])
            X_i = [int(x.split(':')[0]) for x in fields[1:]]
            D_i = [int(x.split(':')[1]) for x in fields[1:]]
            y.append(y_i)
            X.append(X_i)
            D.append(D_i)
            # todo fields=['0', '1:1', '24:1', '127986:1', '131294:1', '131617:1', '131668:1', '131963:1', '133017:1', '133431:1', '133433:1', '133435:1', '133439:1', '133441:1', '133445:1', '133452:1', '133461:1'];
            #  X=[[1, 24, 127986, 131294, 131617, 131668, 131963, 133017, 133431, 133433, 133435, 133439, 133441, 133445, 133452, 133461],...,]
            #  D=[[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
            # print(f"fields={fields}; X={X}; D={D}")
            # break
    y = np.reshape(np.array(y), [-1])
    X = libsvm_2_coo(zip(X, D), (len(X), INPUT_DIM)).tocsr()
    return X, y


def shuffle(data):
    X, y = data
    ind = np.arange(X.shape[0])
    for i in range(7):
        np.random.shuffle(ind)
    return X[ind], y[ind]


def libsvm_2_coo(libsvm_data, shape):
    coo_rows = []
    coo_cols = []
    coo_data = []
    n = 0
    for x, d in libsvm_data:
        coo_rows.extend([n] * len(x))
        coo_cols.extend(x)
        coo_data.extend(d)
        n += 1
        # todo x=[1, 24, 127986, 131294, 131617, 131668, 131963, 133017, 133431, 133433, 133435, 133439, 133441, 133445, 133452, 133461];
        #  coo_rows=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        #  coo_cols=[1, 24, 127986, 131294, 131617, 131668, 131963, 133017, 133431, 133433, 133435, 133439, 133441, 133445, 133452, 133461];
        #  coo_data=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        # print(f"x={x}; coo_rows={coo_rows}; coo_cols={coo_cols}; coo_data={coo_data}")
        # break
    coo_rows = np.array(coo_rows)
    coo_cols = np.array(coo_cols)
    coo_data = np.array(coo_data)
    # todo 当对离散数据进行拟合预测时，往往要对特征进行onehot处理，但onehot是高度稀疏的向量，如果使用List或其他常规的存储方式，对内存占用极大。这时稀疏矩阵类型 coo_matrix / csr_matrix 就派上用场了！
    #  这两种稀疏矩阵类型csr_matrix存储密度更大，但不易手工构建。coo_matrix存储密度相对小，但易于手工构建，常用方法为先手工构建coo_matrix，如果对内存要求高则使用 tocsr() 方法把coo_matrix转换为csr_matrix类型。
    #  分别定义有那些非零元素，以及各个非零元素对应的row和col，最后定义稀疏矩阵的shape
    return coo_matrix((coo_data, (coo_rows, coo_cols)), shape=shape)


def csr_2_input(csr_mat):
    if not isinstance(csr_mat, list):
        # todo 将矩阵转化为 COOrdinate format
        coo_mat = csr_mat.tocoo()
        indices = np.vstack((coo_mat.row, coo_mat.col)).transpose()
        values = csr_mat.data
        shape = csr_mat.shape
        return indices, values, shape
    else:
        # todo 如果csr_mat是列表，列表中的元素总会有不是列表的时候，这时候就将将矩阵转坏额 tocoo() 并且进行后面的计算
        inputs = []
        for csr_i in csr_mat:
            inputs.append(csr_2_input(csr_i))
        return inputs


def slice(csr_data, start=0, size=-1):
    # todo csr_data=([<312437x25 sparse matrix of type '<class 'numpy.int64'>'
    # 	with 525452 stored elements in Compressed Sparse Row format>, <312437x131235 sparse matrix of type '<class 'numpy.int64'>'
    # 	with 411854 stored elements in Compressed Sparse Row format>, <312437x35 sparse matrix of type '<class 'numpy.int64'>'
    # 	with 248323 stored elements in Compressed Sparse Row format>, <312437x367 sparse matrix of type '<class 'numpy.int64'>'
    # 	with 358457 stored elements in Compressed Sparse Row format>, <312437x2 sparse matrix of type '<class 'numpy.int64'>'
    # 	with 9246 stored elements in Compressed Sparse Row format>, <312437x800 sparse matrix of type '<class 'numpy.int64'>'
    # 	with 369456 stored elements in Compressed Sparse Row format>, <312437x961 sparse matrix of type '<class 'numpy.int64'>'
    # 	with 574078 stored elements in Compressed Sparse Row format>, <312437x2 sparse matrix of type '<class 'numpy.int64'>'
    # 	with 49 stored elements in Compressed Sparse Row format>, <312437x2 sparse matrix of type '<class 'numpy.int64'>'
    # 	with 2580 stored elements in Compressed Sparse Row format>, <312437x4 sparse matrix of type '<class 'numpy.int64'>'
    # 	with 312438 stored elements in Compressed Sparse Row format>, <312437x2 sparse matrix of type '<class 'numpy.int64'>'
    # 	with 312437 stored elements in Compressed Sparse Row format>, <312437x4 sparse matrix of type '<class 'numpy.int64'>'
    # 	with 312437 stored elements in Compressed Sparse Row format>, <312437x2 sparse matrix of type '<class 'numpy.int64'>'
    # 	with 312437 stored elements in Compressed Sparse Row format>, <312437x14 sparse matrix of type '<class 'numpy.int64'>'
    # 	with 936708 stored elements in Compressed Sparse Row format>, <312437x5 sparse matrix of type '<class 'numpy.int64'>'
    # 	with 603 stored elements in Compressed Sparse Row format>, <312437x5 sparse matrix of type '<class 'numpy.int64'>'
    # 	with 312437 stored elements in Compressed Sparse Row format>], array([0, 0, 0, ..., 0, 0, 0]));
    # 	csr_data[0]=[<312437x25 sparse matrix of type '<class 'numpy.int64'>'
    # 	with 525452 stored elements in Compressed Sparse Row format>, <312437x131235 sparse matrix of type '<class 'numpy.int64'>'
    # 	with 411854 stored elements in Compressed Sparse Row format>, <312437x35 sparse matrix of type '<class 'numpy.int64'>'
    # 	with 248323 stored elements in Compressed Sparse Row format>, <312437x367 sparse matrix of type '<class 'numpy.int64'>'
    # 	with 358457 stored elements in Compressed Sparse Row format>, <312437x2 sparse matrix of type '<class 'numpy.int64'>'
    # 	with 9246 stored elements in Compressed Sparse Row format>, <312437x800 sparse matrix of type '<class 'numpy.int64'>'
    # 	with 369456 stored elements in Compressed Sparse Row format>, <312437x961 sparse matrix of type '<class 'numpy.int64'>'
    # 	with 574078 stored elements in Compressed Sparse Row format>, <312437x2 sparse matrix of type '<class 'numpy.int64'>'
    # 	with 49 stored elements in Compressed Sparse Row format>, <312437x2 sparse matrix of type '<class 'numpy.int64'>'
    # 	with 2580 stored elements in Compressed Sparse Row format>, <312437x4 sparse matrix of type '<class 'numpy.int64'>'
    # 	with 312438 stored elements in Compressed Sparse Row format>, <312437x2 sparse matrix of type '<class 'numpy.int64'>'
    # 	with 312437 stored elements in Compressed Sparse Row format>, <312437x4 sparse matrix of type '<class 'numpy.int64'>'
    # 	with 312437 stored elements in Compressed Sparse Row format>, <312437x2 sparse matrix of type '<class 'numpy.int64'>'
    # 	with 312437 stored elements in Compressed Sparse Row format>, <312437x14 sparse matrix of type '<class 'numpy.int64'>'
    # 	with 936708 stored elements in Compressed Sparse Row format>, <312437x5 sparse matrix of type '<class 'numpy.int64'>'
    # 	with 603 stored elements in Compressed Sparse Row format>, <312437x5 sparse matrix of type '<class 'numpy.int64'>'
    # 	with 312437 stored elements in Compressed Sparse Row format>]
    print(f"csr_data={csr_data}; csr_data[0]={csr_data[0]}")
    if not isinstance(csr_data[0], list):
        if size == -1 or start + size >= csr_data[0].shape[0]:
            slc_data = csr_data[0][start:]
            slc_labels = csr_data[1][start:]
        else:
            slc_data = csr_data[0][start:start + size]
            slc_labels = csr_data[1][start:start + size]
    else:
        if size == -1 or start + size >= csr_data[0][0].shape[0]:
            slc_data = []
            for d_i in csr_data[0]:
                slc_data.append(d_i[start:])
            slc_labels = csr_data[1][start:]
        else:
            slc_data = []
            for d_i in csr_data[0]:
                slc_data.append(d_i[start:start + size])
            slc_labels = csr_data[1][start:start + size]
    return csr_2_input(slc_data), slc_labels


def split_data(data, skip_empty=True):
    fields = []
    for i in range(len(FIELD_OFFSETS) - 1):
        start_ind = FIELD_OFFSETS[i]
        end_ind = FIELD_OFFSETS[i + 1]
        if skip_empty and start_ind == end_ind:
            continue
        field_i = data[0][:, start_ind:end_ind]
        fields.append(field_i)
    fields.append(data[0][:, FIELD_OFFSETS[-1]:])
    return fields, data[1]


def init_var_map(init_vars, init_path=None):
    if init_path is not None:
        load_var_map = pkl.load(open(init_path, 'rb'))
        print('load variable map from', init_path, load_var_map.keys())
    var_map = {}
    for var_name, var_shape, init_method, dtype in init_vars:
        if init_method == 'zero':
            var_map[var_name] = tf.Variable(tf.zeros(var_shape, dtype=dtype), name=var_name, dtype=dtype)
        elif init_method == 'one':
            var_map[var_name] = tf.Variable(tf.ones(var_shape, dtype=dtype), name=var_name, dtype=dtype)
        elif init_method == 'normal':
            var_map[var_name] = tf.Variable(tf.random_normal(var_shape, mean=0.0, stddev=STDDEV, dtype=dtype),
                                            name=var_name, dtype=dtype)
        elif init_method == 'tnormal':
            var_map[var_name] = tf.Variable(tf.truncated_normal(var_shape, mean=0.0, stddev=STDDEV, dtype=dtype),
                                            name=var_name, dtype=dtype)
        elif init_method == 'uniform':
            var_map[var_name] = tf.Variable(tf.random_uniform(var_shape, minval=MINVAL, maxval=MAXVAL, dtype=dtype),
                                            name=var_name, dtype=dtype)
        elif init_method == 'xavier':
            maxval = np.sqrt(6. / np.sum(var_shape))
            minval = -maxval
            var_map[var_name] = tf.Variable(tf.random_uniform(var_shape, minval=minval, maxval=maxval, dtype=dtype),
                                            name=var_name, dtype=dtype)
        elif isinstance(init_method, int) or isinstance(init_method, float):
            var_map[var_name] = tf.Variable(tf.ones(var_shape, dtype=dtype) * init_method, name=var_name, dtype=dtype)
        elif init_method in load_var_map:
            if load_var_map[init_method].shape == tuple(var_shape):
                var_map[var_name] = tf.Variable(load_var_map[init_method], name=var_name, dtype=dtype)
            else:
                print('BadParam: init method', init_method, 'shape', var_shape, load_var_map[init_method].shape)
        else:
            print('BadParam: init method', init_method)
    return var_map


def activate(weights, activation_function):
    # todo 这里 weights 是和x相乘后的数值
    if activation_function == 'sigmoid':
        return tf.nn.sigmoid(weights)
    elif activation_function == 'softmax':
        return tf.nn.softmax(weights)
    elif activation_function == 'relu':
        return tf.nn.relu(weights)
    elif activation_function == 'tanh':
        return tf.nn.tanh(weights)
    elif activation_function == 'elu':
        return tf.nn.elu(weights)
    elif activation_function == 'none':
        return weights
    else:
        return weights


def get_optimizer(opt_algo, learning_rate, loss):
    if opt_algo == 'adaldeta':
        return tf.train.AdadeltaOptimizer(learning_rate).minimize(loss)
    elif opt_algo == 'adagrad':
        return tf.train.AdagradOptimizer(learning_rate).minimize(loss)
    elif opt_algo == 'adam':
        return tf.train.AdamOptimizer(learning_rate).minimize(loss)
    elif opt_algo == 'ftrl':
        return tf.train.FtrlOptimizer(learning_rate).minimize(loss)
    elif opt_algo == 'gd':
        return tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
    elif opt_algo == 'padagrad':
        return tf.train.ProximalAdagradOptimizer(learning_rate).minimize(loss)
    elif opt_algo == 'pgd':
        return tf.train.ProximalGradientDescentOptimizer(learning_rate).minimize(loss)
    elif opt_algo == 'rmsprop':
        return tf.train.RMSPropOptimizer(learning_rate).minimize(loss)
    else:
        return tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)


def gather_2d(params, indices):
    shape = tf.shape(params)
    flat = tf.reshape(params, [-1])
    flat_idx = indices[:, 0] * shape[1] + indices[:, 1]
    flat_idx = tf.reshape(flat_idx, [-1])
    return tf.gather(flat, flat_idx)


def gather_3d(params, indices):
    shape = tf.shape(params)
    flat = tf.reshape(params, [-1])
    flat_idx = indices[:, 0] * shape[1] * shape[2] + indices[:, 1] * shape[2] + indices[:, 2]
    flat_idx = tf.reshape(flat_idx, [-1])
    return tf.gather(flat, flat_idx)


def gather_4d(params, indices):
    shape = tf.shape(params)
    flat = tf.reshape(params, [-1])
    flat_idx = indices[:, 0] * shape[1] * shape[2] * shape[3] + \
               indices[:, 1] * shape[2] * shape[3] + indices[:, 2] * shape[3] + indices[:, 3]
    flat_idx = tf.reshape(flat_idx, [-1])
    return tf.gather(flat, flat_idx)


def max_pool_2d(params, k):
    _, indices = tf.nn.top_k(params, k, sorted=False)
    shape = tf.shape(indices)
    r1 = tf.reshape(tf.range(shape[0]), [-1, 1])
    r1 = tf.tile(r1, [1, k])
    r1 = tf.reshape(r1, [-1, 1])
    indices = tf.concat([r1, tf.reshape(indices, [-1, 1])], 1)
    return tf.reshape(gather_2d(params, indices), [-1, k])


def max_pool_3d(params, k):
    _, indices = tf.nn.top_k(params, k, sorted=False)
    shape = tf.shape(indices)
    r1 = tf.reshape(tf.range(shape[0]), [-1, 1])
    r2 = tf.reshape(tf.range(shape[1]), [-1, 1])
    r1 = tf.tile(r1, [1, k * shape[1]])
    r2 = tf.tile(r2, [1, k])
    r1 = tf.reshape(r1, [-1, 1])
    r2 = tf.tile(tf.reshape(r2, [-1, 1]), [shape[0], 1])
    indices = tf.concat([r1, r2, tf.reshape(indices, [-1, 1])], 1)
    return tf.reshape(gather_3d(params, indices), [-1, shape[1], k])


def max_pool_4d(params, k):
    _, indices = tf.nn.top_k(params, k, sorted=False)
    shape = tf.shape(indices)
    r1 = tf.reshape(tf.range(shape[0]), [-1, 1])
    r2 = tf.reshape(tf.range(shape[1]), [-1, 1])
    r3 = tf.reshape(tf.range(shape[2]), [-1, 1])
    r1 = tf.tile(r1, [1, shape[1] * shape[2] * k])
    r2 = tf.tile(r2, [1, shape[2] * k])
    r3 = tf.tile(r3, [1, k])
    r1 = tf.reshape(r1, [-1, 1])
    r2 = tf.tile(tf.reshape(r2, [-1, 1]), [shape[0], 1])
    r3 = tf.tile(tf.reshape(r3, [-1, 1]), [shape[0] * shape[1], 1])
    indices = tf.concat([r1, r2, r3, tf.reshape(indices, [-1, 1])], 1)
    return tf.reshape(gather_4d(params, indices), [-1, shape[1], shape[2], k])


if __name__ == '__main__':
    train_file = '../data/train.txt'
    test_file = '../data/test.txt'
    read_data(train_file)
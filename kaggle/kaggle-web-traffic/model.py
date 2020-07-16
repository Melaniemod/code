import tensorflow as tf

import tensorflow.contrib.cudnn_rnn as cudnn_rnn
import tensorflow.contrib.rnn as rnn
import tensorflow.contrib.layers as layers
from tensorflow.python.util import nest

from input_pipe import InputPipe, ModelMode

GRAD_CLIP_THRESHOLD = 10
RNN = cudnn_rnn.CudnnGRU
# RNN = tf.contrib.cudnn_rnn.CudnnLSTM
# RNN = tf.contrib.cudnn_rnn.CudnnRNNRelu


def default_init(seed):
    # replica of tf.glorot_uniform_initializer(seed=seed)
    return layers.variance_scaling_initializer(factor=1.0,
                                               mode="FAN_AVG",
                                               uniform=True,
                                               seed=seed)


def selu(x):
    """
    SELU activation
    https://arxiv.org/abs/1706.02515
    :param x:
    :return:
    """
    with tf.name_scope('elu') as scope:
        alpha = 1.6732632423543772848170429916717
        scale = 1.0507009873554804934193349852946
        return scale * tf.where(x >= 0.0, x, alpha * tf.nn.elu(x))


def make_encoder(time_inputs, encoder_features_depth, is_train, hparams, seed, transpose_output=True):
    """
    Builds encoder, using CUDA RNN
    :param time_inputs: Input tensor, shape [batch, time, features]
    :param encoder_features_depth: Static size for features dimension
    :param is_train:
    :param hparams:
    :param seed:
    :param transpose_output: Transform RNN output to batch-first shape
    :return:
    """

    def build_rnn():
        # todo num_units这里为何叫rnn的深度呢？如何理解深度，不是只有一层吗？num_units 可以理解为每一个时间步骤输出结果的维度 https://zhuanlan.zhihu.com/p/73965801
        #  单层lstm的神经元总数量 = 4 * ((num_units + n) * num_units + num_units；其中每一个时间步骤，输入一个维度为 n 的向量
        return RNN(num_layers=hparams.encoder_rnn_layers, num_units=hparams.rnn_depth,
                   #input_size=encoder_features_depth,
                   kernel_initializer=tf.initializers.random_uniform(minval=-0.05, maxval=0.05,
                                                                      seed=seed + 1 if seed else None),
                   direction='unidirectional',
                   dropout=hparams.encoder_dropout if is_train else 0, seed=seed)

    cuda_model = build_rnn()
    # todo cuda_model <class 'tensorflow.contrib.cudnn_rnn.python.layers.cudnn_rnn.CudnnGRU'>
    #  <tensorflow.contrib.cudnn_rnn.python.layers.cudnn_rnn.CudnnGRU object at 0x7f98a005c668>
    # print("cuda_model",type(cuda_model),cuda_model)

    # [batch, time, features] -> [time, batch, features]
    time_first = tf.transpose(time_inputs, [1, 0, 2])
    rnn_time_input = time_first
    if RNN == tf.contrib.cudnn_rnn.CudnnLSTM:
        # print("LSTM")
        rnn_out, (rnn_state, c_state) = cuda_model(inputs=rnn_time_input)
    else:
        # print("LSTM else")
        # todo output：输出顺序(输出顺序是什么？)；output_h：h的最终状态--(h的最终状态，通过下面的输出感觉像最后一个时间步输出的h；attention用的不是h吧？)
        rnn_out, (rnn_state,) = cuda_model(inputs=rnn_time_input)
        # print("LSTM else -- 2")
        c_state = None
    if transpose_output:
        rnn_out = tf.transpose(rnn_out, [1, 0, 2])
    # todo False; rnn_out=Tensor("m_0/cudnn_gru/CudnnRNN:0", shape=(283, ?, 267), dtype=float32, device=/device:GPU:0);
    #  rnn_state=Tensor("m_0/cudnn_gru/CudnnRNN:1", shape=(1, ?, 267), dtype=float32, device=/device:GPU:0),
    #  c_state=None;
    # print(f"{RNN == tf.contrib.cudnn_rnn.CudnnLSTM}; rnn_out={rnn_out}; rnn_state={rnn_state}; c_state={c_state}; ")
    return rnn_out, rnn_state, c_state


def compressed_readout(rnn_out, hparams, dropout, seed):
    """
    FC compression layer, reduces RNN output depth to hparams.attention_depth
    :param rnn_out:
    :param hparams:
    :param dropout:
    :param seed:
    :return:
    """
    if dropout < 1.0:
        rnn_out = tf.nn.dropout(rnn_out, dropout, seed=seed)
    # todo 为什么需要对 encoder_output 还要整理成另一种 depth
    return tf.layers.dense(rnn_out, hparams.attention_depth,
                           use_bias=True,
                           activation=selu,
                           kernel_initializer=layers.variance_scaling_initializer(factor=1.0, seed=seed),
                           name='compress_readout'
                           )


def make_fingerprint(x, is_train, fc_dropout, seed):
    """
    Calculates 'fingerprint' of timeseries, to feed into attention layer
    :param x:
    :param is_train:
    :param fc_dropout:
    :param seed:
    :return:
    """
    with tf.variable_scope("fingerpint"):
        # x = tf.expand_dims(x, -1)
        # todo 这里是每个样本都有283天的历史数据作为输入，然后每天输入有5维特征对吧？
        with tf.variable_scope('convnet', initializer=layers.variance_scaling_initializer(seed=seed)):
            # todo tf.layers.conv1d()：一维卷积常用于序列数据，如自然语言处理领域：https://www.cnblogs.com/szxspark/p/8445406.html
            #  inputs：张量数据输入，一般是[batch, width, length]
            #  filters：整数，输出空间的维度，可以理解为卷积核(滤波器)的个数
            #  kernel_size：单个整数或元组/列表，指定1D(一维，一行或者一列)卷积窗口的长度。
            #  strides：单个整数或元组/列表，指定卷积的步长，默认为1
            #  padding："SAME" or "VALID" (不区分大小写)是否用0填充，SAME用0填充；VALID不使用0填充，舍去不匹配的多余项。
            c11 = tf.layers.conv1d(x, filters=16, kernel_size=7, activation=tf.nn.relu, padding='same')
            c12 = tf.layers.conv1d(c11, filters=16, kernel_size=3, activation=tf.nn.relu, padding='same')
            # todo  tf.layers.max_pooling1d
            #  inputs：池的张量,秩必须为3.
            #  pool_size：单个整数的整数或元组/列表,表示池化窗口的大小.
            #  strides：单个整数的整数或元组/列表,指定池操作的步幅.
            #  padding：一个字符串,表示填充方法,可以是“valid”或“same”,不区分大小写.
            #  tf.layers.max_pooling2d(): inputs：池的张量,秩必须为4.
            pool1 = tf.layers.max_pooling1d(c12, 2, 2, padding='same')
            c21 = tf.layers.conv1d(pool1, filters=32, kernel_size=3, activation=tf.nn.relu, padding='same')
            c22 = tf.layers.conv1d(c21, filters=32, kernel_size=3, activation=tf.nn.relu, padding='same')
            pool2 = tf.layers.max_pooling1d(c22, 2, 2, padding='same')
            c31 = tf.layers.conv1d(pool2, filters=64, kernel_size=3, activation=tf.nn.relu, padding='same')
            c32 = tf.layers.conv1d(c31, filters=64, kernel_size=3, activation=tf.nn.relu, padding='same')
            pool3 = tf.layers.max_pooling1d(c32, 2, 2, padding='same')
            dims = pool3.shape.dims
            pool3 = tf.reshape(pool3, [-1, dims[1].value * dims[2].value])
            if is_train and fc_dropout < 1.0:
                cnn_out = tf.nn.dropout(pool3, fc_dropout, seed=seed)
            else:
                cnn_out = pool3
        with tf.variable_scope('fc_convnet',
                               initializer=layers.variance_scaling_initializer(factor=1.0, mode='FAN_IN', seed=seed)):
            fc_encoder = tf.layers.dense(cnn_out, 512, activation=selu, name='fc_encoder')
            out_encoder = tf.layers.dense(fc_encoder, 16, activation=selu, name='out_encoder')
            # todo
            #  x:Tensor("m_0/concat:0", shape=(?, 283, 5), dtype=float32, device=/device:GPU:0),
            #  c11:Tensor("m_0/fingerpint/convnet/conv1d/Relu:0", shape=(?, 283, 16), dtype=float32, device=/device:GPU:0)
            #  c12=Tensor("m_0/fingerpint/convnet/conv1d_1/Relu:0", shape=(?, 283, 16), dtype=float32, device=/device:GPU:0);
            #  pool1=Tensor("m_1/fingerpint/convnet/max_pooling1d/Squeeze:0", shape=(?, 142, 16), dtype=float32, device=/device:GPU:0);
            #  c32=Tensor("m_1/fingerpint/convnet/conv1d_5/Relu:0", shape=(?, 71, 64), dtype=float32, device=/device:GPU:0)
            #  pool3=Tensor("m_2/fingerpint/convnet/Reshape:0", shape=(?, 2304), dtype=float32, device=/device:GPU:0)
            #  cnn_out: Tensor("m_2/fingerpint/convnet/dropout/mul_1:0", shape=(?, 2304), dtype=float32, device=/device:GPU:0),
            #  fc_encoder: Tensor("m_2/fingerpint/fc_convnet/fc_encoder/elu/mul_1:0", shape=(?, 512), dtype=float32, device=/device:GPU:0),
            #  out_encoder: Tensor("m_2/fingerpint/fc_convnet/out_encoder/elu/mul_1:0", shape=(?, 16), dtype=float32, device=/device:GPU:0);
            # print(f"x:{x},c11:{c11}, c12={c12}; pool1={pool1};c32={c32}; pool3={pool3}; cnn_out: {cnn_out}, fc_encoder: {fc_encoder}, out_encoder: {out_encoder}; ")
    return out_encoder


def attn_readout_v3(readout, attn_window, attn_heads, page_features, seed):
    # input: [n_days, batch, readout_depth]
    # [n_days, batch, readout_depth] -> [batch(readout_depth), width=n_days, channels=batch]
    readout = tf.transpose(readout, [2, 0, 1])
    # [batch(readout_depth), width, channels] -> [batch, height=1, width, channels]
    # todo tf.newaxis的功能与np.newaxis的功能、用法相同，是增加维度的
    inp = readout[:, tf.newaxis, :, :]

    # todo readout=Tensor("m_0/transpose_1:0", shape=(64, 283, ?), dtype=float32, device=/device:GPU:0),
    #  attn_window=221,
    #  attn_heads=1,
    #  page_features=Tensor("m_0/fingerpint/fc_convnet/out_encoder/elu/mul_1:0", shape=(?, 16), dtype=float32, device=/device:GPU:0), seed)
    # attn_window = train_window - predict_window + 1
    # [batch, attn_window * n_heads]
    # print(f"readout={readout}, attn_window={attn_window}, attn_heads={attn_heads}, page_features={page_features}, seed)")
    filter_logits = tf.layers.dense(page_features, attn_window * attn_heads, name="attn_focus",
                                    kernel_initializer=default_init(seed)
                                    # kernel_initializer=layers.variance_scaling_initializer(uniform=True)
                                    # activation=selu,
                                    # kernel_initializer=layers.variance_scaling_initializer(factor=1.0, mode='FAN_IN')
                                    )
    # [batch, attn_window * n_heads] -> [batch, attn_window, n_heads]
    filter_logits = tf.reshape(filter_logits, [-1, attn_window, attn_heads])

    # attns_max = tf.nn.softmax(filter_logits, dim=1)
    attns_max = filter_logits / tf.reduce_sum(filter_logits, axis=1, keep_dims=True)
    # [batch, attn_window, n_heads] -> [width(attn_window), channels(batch), n_heads]
    attns_max = tf.transpose(attns_max, [1, 0, 2])

    # [width(attn_window), channels(batch), n_heads] -> [height(1), width(attn_window), channels(batch), multiplier(n_heads)]
    attn_filter = attns_max[tf.newaxis, :, :, :]
    # [batch(readout_depth), height=1, width=n_days, channels=batch] -> [batch(readout_depth), height=1, width=predict_window, channels=batch*n_heads]
    # todo tf.nn.depthwise_conv2d(input,filter,strides,padding,rate=None,name=None,data_format=None)
    #  https://blog.csdn.net/mao_xiao_feng/article/details/78003476
    #  在给定 4-Dinput和filter张量的情况下计算 2-D 深度卷积.可以辅助实现分组卷积。
    #  input：指需要做卷积的输入图像，要求是一个4维Tensor，shape为[batch, in_height, in_width, in_channels],
    #  filter：相当于CNN中的卷积核，要求是一个4维Tensor，具有[filter_height, filter_width, in_channels, channel_multiplier]这样的shape
    #  strides：卷积的滑动步长。
    #  返回一个Tensor，shape为[batch, out_height, out_width, in_channels * channel_multiplier]
    #  inp=Tensor("m_0/strided_slice:0", shape=(64, 1, 283, ?), dtype=float32, device=/device:GPU:0),
    #  attn_filter=Tensor("m_0/strided_slice_1:0", shape=(1, 221, ?, 1), dtype=float32, device=/device:GPU:0)
    # print(f"inp={inp}, attn_filter={attn_filter}")
    averaged = tf.nn.depthwise_conv2d_native(inp, attn_filter, [1, 1, 1, 1], 'VALID')
    # [batch, height=1, width=predict_window, channels=readout_depth*n_neads] -> [batch(depth), predict_window, batch*n_heads]
    # todo 从tensor中删除所有大小是1的维度
    attn_features = tf.squeeze(averaged, 1)
    # [batch(depth), predict_window, batch*n_heads] -> [batch*n_heads, predict_window, depth]
    attn_features = tf.transpose(attn_features, [2, 1, 0])
    # [batch * n_heads, predict_window, depth] -> n_heads * [batch, predict_window, depth]
    # todo ::的完整写法为 seq[start:end:step]
    heads = [attn_features[head_no::attn_heads] for head_no in range(attn_heads)]
    # todo attn_features=Tensor("m_0/transpose_3:0", shape=(?, 63, 64), dtype=float32, device=/device:GPU:0);
    #  heads=[<tf.Tensor 'm_0/strided_slice_2:0' shape=(?, 63, 64) dtype=float32>]
    # print(f"attn_features={attn_features}; heads={heads}")
    # n_heads * [batch, predict_window, depth] -> [batch, predict_window, depth*n_heads]
    result = tf.concat(heads, axis=-1)
    # attn_diag = tf.unstack(attns_max, axis=-1)
    return result, None


def calc_smape_rounded(true, predicted, weights):
    """
    Calculates SMAPE on rounded submission values. Should be close to official SMAPE in competition
    :param true:
    :param predicted:
    :param weights: Weights mask to exclude some values
    :return:
    """
    n_valid = tf.reduce_sum(weights)
    true_o = tf.round(tf.expm1(true))
    pred_o = tf.maximum(tf.round(tf.expm1(predicted)), 0.0)
    summ = tf.abs(true_o) + tf.abs(pred_o)
    zeros = summ < 0.01
    raw_smape = tf.abs(pred_o - true_o) / summ * 2.0
    smape = tf.where(zeros, tf.zeros_like(summ, dtype=tf.float32), raw_smape)
    return tf.reduce_sum(smape * weights) / n_valid


def smape_loss(true, predicted, weights):
    """
    Differentiable SMAPE loss
    :param true: Truth values
    :param predicted: Predicted values
    :param weights: Weights mask to exclude some values
    :return:
    """
    epsilon = 0.1  # Smoothing factor, helps SMAPE to be well-behaved near zero
    # todo expm1:自然指数减1，即e^x-1
    true_o = tf.expm1(true)
    pred_o = tf.expm1(predicted)
    summ = tf.maximum(tf.abs(true_o) + tf.abs(pred_o) + epsilon, 0.5 + epsilon)
    smape = tf.abs(pred_o - true_o) / summ * 2.0
    # todo 为何算这样一种损失函数呢？SMAPE 对称平均绝对百分比误差（Symmetric Mean Absolute Percentage Error） https://blog.csdn.net/guolindonggld/article/details/87856780
    # todo tf.losses.compute_weighted_loss：返回与losses相同类型的加权损失Tensor,如果reduction是NONE,它的形状与losses相同；否则,它是标量.
    return tf.losses.compute_weighted_loss(smape, weights, loss_collection=None)


def decode_predictions(decoder_readout, inp: InputPipe):
    """
    Converts normalized prediction values to log1p(pageviews), e.g. reverts normalization
    :param decoder_readout: Decoder output, shape [n_days, batch]
    :param inp: Input tensors
    :return:
    """
    # [n_days, batch] -> [batch, n_days]
    batch_readout = tf.transpose(decoder_readout)
    batch_std = tf.expand_dims(inp.norm_std, -1)
    batch_mean = tf.expand_dims(inp.norm_mean, -1)
    return batch_readout * batch_std + batch_mean


def calc_loss(predictions, true_y, additional_mask=None):
    """
    Calculates losses, ignoring NaN true values (assigning zero loss to them)
    :param predictions: Predicted values
    :param true_y: True values
    :param additional_mask:
    :return: MAE loss, differentiable SMAPE loss, competition SMAPE loss
    """
    # Take into account NaN's in true values
    mask = tf.is_finite(true_y)
    # Fill NaNs by zeros (can use any value)
    true_y = tf.where(mask, true_y, tf.zeros_like(true_y))
    # Assign zero weight to NaNs
    weights = tf.to_float(mask)
    if additional_mask is not None:
        weights = weights * tf.expand_dims(additional_mask, axis=0)
    # todo weights作为loss的系数.如果提供了标量,那么损失只是按给定值进行缩放.如果weights是形状为[batch_size]的Tensor,
    #  则批次中每个样品的总loss由weights向量中的相应元素重新调整.如果weights的形状与predictions的形状匹配,
    #  则每个predictions的预测元素的loss由相应的weights值缩放.
    #  返回：该函数返回加权loss浮动Tensor.如果reduction是NONE,则它的形状与labels相同；否则,它是标量.
    mae_loss = tf.losses.absolute_difference(labels=true_y, predictions=predictions, weights=weights)
    return mae_loss, smape_loss(true_y, predictions, weights), calc_smape_rounded(true_y, predictions,
                                                                                  weights), tf.size(true_y)


def make_train_op(loss, ema_decay=None, prefix=None):
    optimizer = tf.train.AdamOptimizer()
    glob_step = tf.train.get_global_step()

    # Add regularization losses
    reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    total_loss = loss + reg_losses if reg_losses else loss

    # Clip gradients -- 梯度修剪（梯度修剪主要避免训练梯度爆炸和消失问题）
    # todo https://blog.csdn.net/lenbow/article/details/52218551    https://blog.csdn.net/NockinOnHeavensDoor/article/details/80632677
    #  梯度修剪的步骤
    #  1、使用函数compute_gradients()计算梯度; 返回一个以元组(gradient, variable)组成的列表
    #  2、按照自己的愿望处理梯度
    #  3、使用函数apply_gradients()应用处理过后的梯度; 返回一个应用指定的梯度的操作Operation。
    grads_and_vars = optimizer.compute_gradients(total_loss)
    gradients, variables = zip(*grads_and_vars)
    # todo https://blog.csdn.net/u013713117/article/details/56281715
    #  tf.clip_by_global_norm(t_list, clip_norm, use_norm=None, name=None)。t_list 是梯度张量， clip_norm 是截取的比率。
    #  这个函数返回：截取过的梯度张量和一个所有张量的全局范数。
    #  更新共公式：t_list[i] * clip_norm / max(global_norm, clip_norm)。global_norm 是所有梯度的平方和，
    #  如果 clip_norm > global_norm ，就不进行截取。但是这个函数的速度比clip_by_norm() 要慢。
    clipped_gradients, glob_norm = tf.clip_by_global_norm(gradients, GRAD_CLIP_THRESHOLD)
    # todo 难道下面对 zip(clipped_gradients, variables) 的处理会影响 variables 的值？应该会影响！这里先影响了variables的值，之后再更新 update_ema
    sgd_op, glob_norm = optimizer.apply_gradients(zip(clipped_gradients, variables)), glob_norm

    # Apply SGD averaging
    # todo ema_decay=0.99; prefix=m_0; 数值型只要数值>0就为True；只要不是''空字符串就为True
    # print(f"ema_decay={ema_decay}; prefix={prefix}")
    if ema_decay:
        # todo https://blog.csdn.net/UESTC_C2_403/article/details/72235334
        #  ExponentialMovingAverage 对参数采用滑动平均的方法更新。
        ema = tf.train.ExponentialMovingAverage(decay=ema_decay, num_updates=glob_step)
        if prefix:
            # Some magic to handle multiple models trained in single graph
            ema_vars = [var for var in variables if var.name.startswith(prefix)]
            # todo ema_vars=[<tf.Variable 'm_0/cudnn_gru/opaque_kernel:0' shape=<unknown> dtype=float32_ref>,
            #  <tf.Variable 'm_0/compress_readout/kernel:0' shape=(267, 64) dtype=float32_ref>,
            #  <tf.Variable 'm_0/compress_readout/bias:0' shape=(64,) dtype=float32_ref>,
            #  <tf.Variable 'm_0/fingerpint/convnet/conv1d/kernel:0' shape=(7, 5, 16) dtype=float32_ref>,
            #  <tf.Variable 'm_0/fingerpint/convnet/conv1d/bias:0' shape=(16,) dtype=float32_ref>,
            #  <tf.Variable 'm_0/fingerpint/convnet/conv1d_1/kernel:0' shape=(3, 16, 16) dtype=float32_ref>,
            #  <tf.Variable 'm_0/fingerpint/convnet/conv1d_1/bias:0' shape=(16,) dtype=float32_ref>,
            #  <tf.Variable 'm_0/fingerpint/convnet/conv1d_2/kernel:0' shape=(3, 16, 32) dtype=float32_ref>,
            #  <tf.Variable 'm_0/fingerpint/convnet/conv1d_2/bias:0' shape=(32,) dtype=float32_ref>,
            #  <tf.Variable 'm_0/fingerpint/convnet/conv1d_3/kernel:0' shape=(3, 32, 32) dtype=float32_ref>,
            #  <tf.Variable 'm_0/fingerpint/convnet/conv1d_3/bias:0' shape=(32,) dtype=float32_ref>,
            #  <tf.Variable 'm_0/fingerpint/convnet/conv1d_4/kernel:0' shape=(3, 32, 64) dtype=float32_ref>,
            #  <tf.Variable 'm_0/fingerpint/convnet/conv1d_4/bias:0' shape=(64,) dtype=float32_ref>,
            #  <tf.Variable 'm_0/fingerpint/convnet/conv1d_5/kernel:0' shape=(3, 64, 64) dtype=float32_ref>,
            #  <tf.Variable 'm_0/fingerpint/convnet/conv1d_5/bias:0' shape=(64,) dtype=float32_ref>,
            #  <tf.Variable 'm_0/fingerpint/fc_convnet/fc_encoder/kernel:0' shape=(2304, 512) dtype=float32_ref>,
            #  <tf.Variable 'm_0/fingerpint/fc_convnet/fc_encoder/bias:0' shape=(512,) dtype=float32_ref>,
            #  <tf.Variable 'm_0/fingerpint/fc_convnet/out_encoder/kernel:0' shape=(512, 16) dtype=float32_ref>,
            #  <tf.Variable 'm_0/fingerpint/fc_convnet/out_encoder/bias:0' shape=(16,) dtype=float32_ref>,
            #  <tf.Variable 'm_0/attn_focus/kernel:0' shape=(16, 221) dtype=float32_ref>,
            #  <tf.Variable 'm_0/attn_focus/bias:0' shape=(221,) dtype=float32_ref>,
            #  <tf.Variable 'm_0/gru_cell/w_ru:0' shape=(291, 534) dtype=float32_ref>,
            #  <tf.Variable 'm_0/gru_cell/b_ru:0' shape=(534,) dtype=float32_ref>,
            #  <tf.Variable 'm_0/gru_cell/w_c:0' shape=(291, 267) dtype=float32_ref>,
            #  <tf.Variable 'm_0/gru_cell/b_c:0' shape=(267,) dtype=float32_ref>,
            #  <tf.Variable 'm_0/decoder_output_proj/kernel:0' shape=(267, 1) dtype=float32_ref>,
            #  <tf.Variable 'm_0/decoder_output_proj/bias:0' shape=(1,) dtype=float32_ref>]
            # print(f"ema_vars={ema_vars}")
        else:
            ema_vars = variables
        # todo tf.train.ExponentialMovingAverage.apply(var_list=None)	对var_list变量保持移动平均
        update_ema = ema.apply(ema_vars)
        with tf.control_dependencies([sgd_op]):
            #  todo tf.group(*inputs,**kwargs) 其中*inputs是一个一个operation，一旦ops完成了，那么传入的tensor1,tensor2,...等都会完成，
            #   经常用于组合一些训练节点。tf.group()返回的是个操作，而不是值
            #   这里反向传播首先计算了他们的梯度，然后做一些修剪，然后再指数平滑最终更新参数？
            training_op = tf.group(update_ema)
    else:
        training_op = sgd_op
        ema = None
    # todo update_ema=name: "m_2/ExponentialMovingAverage"
    #  op: "NoOp"
    #  input: "^m_2/ExponentialMovingAverage/AssignMovingAvg"
    #  input: "^m_2/ExponentialMovingAverage/AssignMovingAvg_1"
    #  input: "^m_2/ExponentialMovingAverage/AssignMovingAvg_2"
    #  input: "^m_2/ExponentialMovingAverage/AssignMovingAvg_3"
    #  input: "^m_2/ExponentialMovingAverage/AssignMovingAvg_4"
    #  input: "^m_2/ExponentialMovingAverage/AssignMovingAvg_5"
    #  input: "^m_2/ExponentialMovingAverage/AssignMovingAvg_6"
    #  input: "^m_2/ExponentialMovingAverage/AssignMovingAvg_7"
    #  input: "^m_2/ExponentialMovingAverage/AssignMovingAvg_8"
    #  input: "^m_2/ExponentialMovingAverage/AssignMovingAvg_9"
    #  input: "^m_2/ExponentialMovingAverage/AssignMovingAvg_10"
    #  input: "^m_2/ExponentialMovingAverage/AssignMovingAvg_11"
    #  input: "^m_2/ExponentialMovingAverage/AssignMovingAvg_12"
    #  input: "^m_2/ExponentialMovingAverage/AssignMovingAvg_13"
    #  input: "^m_2/ExponentialMovingAverage/AssignMovingAvg_14"
    #  input: "^m_2/ExponentialMovingAverage/AssignMovingAvg_15"
    #  input: "^m_2/ExponentialMovingAverage/AssignMovingAvg_16"
    #  input: "^m_2/ExponentialMovingAverage/AssignMovingAvg_17"
    #  input: "^m_2/ExponentialMovingAverage/AssignMovingAvg_18"
    #  input: "^m_2/ExponentialMovingAverage/AssignMovingAvg_19"
    #  input: "^m_2/ExponentialMovingAverage/AssignMovingAvg_20"
    #  input: "^m_2/ExponentialMovingAverage/AssignMovingAvg_21"
    #  input: "^m_2/ExponentialMovingAverage/AssignMovingAvg_22"
    #  input: "^m_2/ExponentialMovingAverage/AssignMovingAvg_23"
    #  input: "^m_2/ExponentialMovingAverage/AssignMovingAvg_24"
    #  input: "^m_2/ExponentialMovingAverage/AssignMovingAvg_25"
    #  input: "^m_2/ExponentialMovingAverage/AssignMovingAvg_26"
    #  device: "/device:GPU:0"
    #  ; training_op=name: "m_2/group_deps"
    #  op: "NoOp"
    #  input: "^m_2/Adam"
    #  input: "^m_2/ExponentialMovingAverage"
    #  device: "/device:GPU:0"
    # print(f"update_ema={update_ema}; training_op={training_op}")
    return training_op, glob_norm, ema


def convert_cudnn_state_v2(h_state, hparams, seed, c_state=None, dropout=1.0):
    """
    Converts RNN state tensor from cuDNN representation to TF RNNCell compatible representation.
    :param h_state: tensor [num_layers, batch_size, depth]
    :param c_state: LSTM additional state, should be same shape as h_state
    :return: TF cell representation matching RNNCell.state_size structure for compatible cell
    """

    def squeeze(seq):
        return tuple(seq) if len(seq) > 1 else seq[0]

    def wrap_dropout(structure):
        if dropout < 1.0:
            # todo map(func, *iterables) 和 nest.map_structure(func, *structure, **check_types_dict)功能类似。
            #  两者都是对一个可循环结构的元素依次应用函数的过程。
            #  不同的是：map（）返回一个map 对象;  map_structure（）返回一个与参数structure有相同arity的structure。
            return nest.map_structure(lambda x: tf.nn.dropout(x, keep_prob=dropout, seed=seed), structure)
        else:
            return structure

    # Cases:
    # decoder_layer = encoder_layers, straight mapping
    # encoder_layers > decoder_layers: get outputs of upper encoder layers
    # encoder_layers < decoder_layers: feed encoder outputs to lower decoder layers, feed zeros to top layers
    # todo tf.stack（）这是一个矩阵拼接的函数，tf.unstack（）则是一个矩阵分解的函数
    h_layers = tf.unstack(h_state)
    print(f"")
    # todo 下面这样一番操作是什么意思？
    #  hparams.encoder_rnn_layers =1; hparams.decoder_rnn_layers=1; hparams.encoder_rnn_layers >= hparams.decoder_rnn_layers=True;
    #  h_state=Tensor("m_0/cudnn_gru/CudnnRNN:1", shape=(1, ?, 267), dtype=float32, device=/device:GPU:0);
    #  h_layers=[<tf.Tensor 'm_0/unstack:0' shape=(?, 267) dtype=float32>]
    #  squeeze(wrap_dropout(h_layers[hparams.encoder_rnn_layers - hparams.decoder_rnn_layers:]))
    #  3=Tensor("m_0/dropout/mul_1:0", shape=(?, 267), dtype=float32, device=/device:GPU:0)
    # print(f"hparams.encoder_rnn_layers ={hparams.encoder_rnn_layers}; hparams.decoder_rnn_layers={hparams.decoder_rnn_layers}; "
    #       f"hparams.encoder_rnn_layers >= hparams.decoder_rnn_layers={hparams.encoder_rnn_layers >= hparams.decoder_rnn_layers}; "
    #       f"h_state={h_state}; h_layers={h_layers}")
    # todo 这里做dropout为什么还要先unstack，dropout之后再squeeze
    if hparams.encoder_rnn_layers >= hparams.decoder_rnn_layers:
        # todo decoder_rnn_layers有几层对encoder的最上面几层做输出。
        return squeeze(wrap_dropout(h_layers[hparams.encoder_rnn_layers - hparams.decoder_rnn_layers:]))
    else:
        lower_inputs = wrap_dropout(h_layers)
        upper_inputs = [tf.zeros_like(h_layers[0]) for _ in
                        range(hparams.decoder_rnn_layers - hparams.encoder_rnn_layers)]
        return squeeze(lower_inputs + upper_inputs)


def rnn_stability_loss(rnn_output, beta):
    """
    # todo 使用这种正则的原因：I tried to use RNN activation regularizations from the paper "Regularizing RNNs by Stabilizing Activations",
       because internal weights in cuDNN GRU can't be directly regularized (or I did not found a right way to do this).
       Stability loss didn't work at all, activation loss gave some very slight improvement for low (1e-06..1e-05) loss weights.
    Regularizing Rnns By Stabilizing Activations
    https://arxiv.org/pdf/1511.08400.pdf
    :param rnn_output: [time, batch, features]
    :return: loss value
    """
    if beta == 0.0:
        return 0.0
    # [time, batch, features] -> [time, batch]
    l2 = tf.sqrt(tf.reduce_sum(tf.square(rnn_output), axis=-1))
    #  [time, batch] -> []
    return beta * tf.reduce_mean(tf.square(l2[1:] - l2[:-1]))


def rnn_activation_loss(rnn_output, beta):
    """
    Regularizing Rnns By Stabilizing Activations
    https://arxiv.org/pdf/1511.08400.pdf
    :param rnn_output: [time, batch, features]
    :return: loss value
    """
    if beta == 0.0:
        return 0.0
    return tf.nn.l2_loss(rnn_output) * beta


class Model:
    def __init__(self, inp: InputPipe, hparams, is_train, seed, graph_prefix=None, asgd_decay=None, loss_mask=None):
        """
        Encoder-decoder prediction model
        :param inp: Input tensors
        :param hparams:
        :param is_train:
        :param seed:
        :param graph_prefix: Subgraph prefix for multi-model graph
        :param asgd_decay: Decay for SGD averaging
        :param loss_mask: Additional mask for losses calculation (one value for each prediction day), shape=[predict_window]
        """
        self.is_train = is_train
        self.inp = inp
        self.hparams = hparams
        self.seed = seed
        # self.inp = inp
        # todo mdoel -- inp=<input_pipe.InputPipe object at 0x7f46e4c98c18>
        # print(f"mdoel -- inp={inp}")
        encoder_output, h_state, c_state = make_encoder(inp.time_x, inp.encoder_features_depth, is_train, hparams, seed,
                                                        transpose_output=False)
        # Encoder activation losses
        enc_stab_loss = rnn_stability_loss(encoder_output, hparams.encoder_stability_loss / inp.train_window)
        enc_activation_loss = rnn_activation_loss(encoder_output, hparams.encoder_activation_loss / inp.train_window)

        # Convert state from cuDNN representation to TF RNNCell-compatible representation
        encoder_state = convert_cudnn_state_v2(h_state, hparams, c_state,
                                               dropout=hparams.gate_dropout if is_train else 1.0)
        # todo encoder_state=Tensor("m_0/dropout/mul_1:0", shape=(?, 267), dtype=float32, device=/device:GPU:0)
        # print(f"encoder_state={encoder_state}")

        # Attention calculations
        # Compress encoder outputs
        enc_readout = compressed_readout(encoder_output, hparams,
                                         dropout=hparams.encoder_readout_dropout if is_train else 1.0, seed=seed)
        # todo encoder_output=Tensor("m_0/cudnn_gru/CudnnRNN:0", shape=(283, ?, 267), dtype=float32, device=/device:GPU:0);
        #  enc_readout=Tensor("m_0/compress_readout/elu/mul_1:0", shape=(283, ?, 64), dtype=float32, device=/device:GPU:0)
        # print(f"encoder_output={encoder_output}; enc_readout={enc_readout}")
        # Calculate fingerprint from input features
        fingerprint_inp = tf.concat([inp.lagged_x, tf.expand_dims(inp.norm_x, -1)], axis=-1)
        fingerprint = make_fingerprint(fingerprint_inp, is_train, hparams.fingerprint_fc_dropout, seed)
        # todo fingerprint_inp=Tensor("m_0/concat:0", shape=(?, 283, 5), dtype=float32, device=/device:GPU:0);
        #  fingerprint=Tensor("m_0/fingerpint/fc_convnet/out_encoder/elu/mul_1:0", shape=(?, 16), dtype=float32, device=/device:GPU:0)
        # print(f"fingerprint_inp={fingerprint_inp}; fingerprint={fingerprint}")
        # Calculate attention vector
        attn_features, attn_weights = attn_readout_v3(enc_readout, inp.attn_window, hparams.attention_heads,
                                                      fingerprint, seed=seed)

        # Run decoder
        decoder_targets, decoder_outputs = self.decoder(encoder_state,
                                                        attn_features if hparams.use_attn else None,
                                                        inp.time_y, inp.norm_x[:, -1])
        # todo decoder_targets=Tensor("m_1/Squeeze_1:0", shape=(?, ?), dtype=float32, device=/device:GPU:0),
        #  decoder_outputs=Tensor("m_1/TensorArrayStack_1/TensorArrayGatherV3:0", shape=(?, ?, 267), dtype=float32)
        # print(f"decoder_targets={decoder_targets}, decoder_outputs={decoder_outputs}")
        # Decoder activation losses
        dec_stab_loss = rnn_stability_loss(decoder_outputs, hparams.decoder_stability_loss / inp.predict_window)
        dec_activation_loss = rnn_activation_loss(decoder_outputs, hparams.decoder_activation_loss / inp.predict_window)

        # Get final denormalized predictions
        self.predictions = decode_predictions(decoder_targets, inp)

        # Calculate losses and build training op
        if inp.mode == ModelMode.PREDICT:
            # Pseudo-apply ema to get variable names later in ema.variables_to_restore()
            # This is copypaste from make_train_op()
            if asgd_decay:
                self.ema = tf.train.ExponentialMovingAverage(decay=asgd_decay)
                # todo 该函数可以用来获取key集合中的所有元素，返回一个列表。列表的顺序依变量放入集合中的先后而定。
                variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
                if graph_prefix:
                    ema_vars = [var for var in variables if var.name.startswith(graph_prefix)]
                else:
                    ema_vars = variables
                # todo he apply() method adds shadow copies of trained variables and add ops that maintain a moving average of the trained variables in their shadow copies. It is used when building the training model. The ops that maintain moving averages are typically run after each training step. The average() and average_name() methods give access to the shadow variables and their names. They are useful when building an evaluation model, or when restoring a model from a checkpoint file. They help use the moving averages in place of the last trained values for evaluations.
                self.ema.apply(ema_vars)
        else:
            self.mae, smape_loss, self.smape, self.loss_item_count = calc_loss(self.predictions, inp.true_y,
                                                                               additional_mask=loss_mask)
            if is_train:
                # Sum all losses
                total_loss = smape_loss + enc_stab_loss + dec_stab_loss + enc_activation_loss + dec_activation_loss
                self.train_op, self.glob_norm, self.ema = make_train_op(total_loss, asgd_decay, prefix=graph_prefix)
        # todo inp.mode == ModelMode.PREDICT=False;is_train=True;self.train_op=name: "m_2/group_deps"
        # print(f"inp.mode == ModelMode.PREDICT={inp.mode == ModelMode.PREDICT};is_train={is_train};self.train_op={self.train_op}")



    def default_init(self, seed_add=0):
        return default_init(self.seed + seed_add)

    def decoder(self, encoder_state, attn_features, prediction_inputs, previous_y):
        # todo encoder_state=Tensor("m_0/dropout/mul_1:0", shape=(?, 267), dtype=float32, device=/device:GPU:0),
        #  attn_features=None,
        #  prediction_inputs=Tensor("m_0/input/IteratorGetNext:5", shape=(?, 63, 23), dtype=float32, device=/device:CPU:0),
        #  previous_y=Tensor("m_0/strided_slice_3:0", shape=(?,), dtype=float32, device=/device:GPU:0)
        # print(f"encoder_state={encoder_state}, attn_features={attn_features}, prediction_inputs={prediction_inputs}, previous_y={previous_y}")
        """
        :param encoder_state: shape [batch_size, encoder_rnn_depth]
        :param prediction_inputs: features for prediction days, tensor[batch_size, time, input_depth]
        :param previous_y: Last day pageviews, shape [batch_size]
        :param attn_features: Additional features from attention layer, shape [batch, predict_window, readout_depth*n_heads]
        :return: decoder rnn output
        """
        hparams = self.hparams

        def build_cell(idx):
            with tf.variable_scope('decoder_cell', initializer=self.default_init(idx)):
                # todo rnn_depth=267; 这里编解码的 num_unit 是相等的；直接输出的output也是相等的，如果要的到时间步的输出，还要在上面加一层全连接层。见函数 project_output
                cell = rnn.GRUBlockCell(self.hparams.rnn_depth)
                has_dropout = hparams.decoder_input_dropout[idx] < 1 \
                              or hparams.decoder_state_dropout[idx] < 1 or hparams.decoder_output_dropout[idx] < 1

                if self.is_train and has_dropout:
                    attn_depth = attn_features.shape[-1].value if attn_features is not None else 0
                    input_size = attn_depth + prediction_inputs.shape[-1].value + 1 if idx == 0 else self.hparams.rnn_depth
                    cell = rnn.DropoutWrapper(cell, dtype=tf.float32, input_size=input_size,
                                              variational_recurrent=hparams.decoder_variational_dropout[idx],
                                              input_keep_prob=hparams.decoder_input_dropout[idx],
                                              output_keep_prob=hparams.decoder_output_dropout[idx],
                                              state_keep_prob=hparams.decoder_state_dropout[idx], seed=self.seed + idx)
                return cell
        # todo decoder_rnn_layers=1
        if hparams.decoder_rnn_layers > 1:
            cells = [build_cell(idx) for idx in range(hparams.decoder_rnn_layers)]
            cell = rnn.MultiRNNCell(cells)
        else:
            cell = build_cell(0)

        nest.assert_same_structure(encoder_state, cell.state_size)
        predict_days = self.inp.predict_window
        assert prediction_inputs.shape[1] == predict_days

        # [batch_size, time, input_depth] -> [time, batch_size, input_depth]
        inputs_by_time = tf.transpose(prediction_inputs, [1, 0, 2])

        # Return raw outputs for RNN losses calculation
        return_raw_outputs = self.hparams.decoder_stability_loss > 0.0 or self.hparams.decoder_activation_loss > 0.0
        # todo self.hparams.decoder_stability_loss=0.0;
        #  self.hparams.decoder_activation_loss=5e-06;
        #  return_raw_outputs=True
        # print(f"self.hparams.decoder_stability_loss={self.hparams.decoder_stability_loss}; self.hparams.decoder_activation_loss={self.hparams.decoder_activation_loss}; return_raw_outputs={return_raw_outputs}")

        # Stop condition for decoding loop
        def cond_fn(time, prev_output, prev_state, array_targets: tf.TensorArray, array_outputs: tf.TensorArray):
            return time < predict_days

        # FC projecting layer to get single predicted value from RNN output
        def project_output(tensor):
            return tf.layers.dense(tensor, 1, name='decoder_output_proj', kernel_initializer=self.default_init())

        def loop_fn(time, prev_output, prev_state, array_targets: tf.TensorArray, array_outputs: tf.TensorArray):
            """
            Main decoder loop
            :param time: Day number
            :param prev_output: Output(prediction) from previous step
            :param prev_state: RNN state tensor from previous step
            :param array_targets: Predictions, each step will append new value to this array
            :param array_outputs: Raw RNN outputs (for regularization losses)
            :return:
            """
            # RNN inputs for current step
            # todo decode 第time个时间步的特征，shape为[batch_size,input_depth]
            features = inputs_by_time[time]

            # [batch, predict_window, readout_depth * n_heads] -> [batch, readout_depth * n_heads]
            # todo attn_features=None
            # print(f"attn_features={attn_features}")
            if attn_features is not None:
                #  [batch_size, 1] + [batch_size, input_depth]
                attn = attn_features[:, time, :]
                # Append previous predicted value + attention vector to input features
                next_input = tf.concat([prev_output, features, attn], axis=1)
            else:
                # Append previous predicted value to input features
                next_input = tf.concat([prev_output, features], axis=1)

            # Run RNN cell
            output, state = cell(next_input, prev_state)
            # Make prediction from RNN outputs
            projected_output = project_output(output)
            # Append step results to the buffer arrays
            if return_raw_outputs:
                array_outputs = array_outputs.write(time, output)
            array_targets = array_targets.write(time, projected_output)
            # Increment time and return
            return time + 1, projected_output, state, array_targets, array_outputs

        # Initial values for loop
        loop_init = [tf.constant(0, dtype=tf.int32),
                     tf.expand_dims(previous_y, -1),
                     encoder_state,
                     tf.TensorArray(dtype=tf.float32, size=predict_days),
                     tf.TensorArray(dtype=tf.float32, size=predict_days) if return_raw_outputs else tf.constant(0)]
        # Run the loop
        # todo https://blog.csdn.net/u011509971/article/details/78805727    https://blog.csdn.net/sinat_34474705/article/details/79402967
        #  tf.while_loop(cond, body, loop_vars) 可以这样理解:while_loop先根据变量列表loop_vars运行cond函数并返回一个布尔型张量，
        #  如果为真，则进入循环体body继续执行并返回一组变量列表的值（与loop_vars个数及类型等要一致，相当于更新了一遍loop_vars），
        #  否则退出while_loop循环并返回最终的变量列表中所有的值。
        _, _, _, targets_ta, outputs_ta = tf.while_loop(cond_fn, loop_fn, loop_init)
        # todo targets_ta=<tensorflow.python.ops.tensor_array_ops.TensorArray object at 0x7fe83057a198>;
        #  outputs_ta=<tensorflow.python.ops.tensor_array_ops.TensorArray object at 0x7fe8306387f0>
        # print(f"targets_ta={targets_ta}; outputs_ta={outputs_ta}")

        # Get final tensors from buffer arrays
        targets = targets_ta.stack()
        # [time, batch_size, 1] -> [time, batch_size]
        targets = tf.squeeze(targets, axis=-1)
        raw_outputs = outputs_ta.stack() if return_raw_outputs else None
        return targets, raw_outputs

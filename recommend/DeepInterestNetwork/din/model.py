import tensorflow as tf

from Dice import dice

class Model(object):

  def __init__(self, user_count, item_count, cate_count, cate_list, predict_batch_size, predict_ads_num):

    # todo 这里 self.u 用户 ID 在构建模型框架以及模型优化的时候并没有用到
    self.u = tf.placeholder(tf.int32, [None,]) # [B]
    self.i = tf.placeholder(tf.int32, [None,]) # [B]
    # todo self.j：用户是否点击第 i 个油站
    self.j = tf.placeholder(tf.int32, [None,]) # [B]
    #
    self.y = tf.placeholder(tf.float32, [None,]) # [B]
    self.hist_i = tf.placeholder(tf.int32, [None, None]) # [B, T]
    self.sl = tf.placeholder(tf.int32, [None,]) # [B]
    self.lr = tf.placeholder(tf.float64, [])

    hidden_units = 128

    # user_emb_w = tf.get_variable("user_emb_w", [user_count, hidden_units])
    # todo 为什么这里 embedding 的大小使用 hidden_unit // 2
    item_emb_w = tf.get_variable("item_emb_w", [item_count, hidden_units // 2])
    item_b = tf.get_variable("item_b", [item_count],
                             initializer=tf.constant_initializer(0.0))
    cate_emb_w = tf.get_variable("cate_emb_w", [cate_count, hidden_units // 2])
    cate_list = tf.convert_to_tensor(cate_list, dtype=tf.int64)

    # todo 首先得到每个商品的类目ID；然后根据类目ID得到 embedding；然后将 item 的 embedding 和类目的 embedding concat;
    ic = tf.gather(cate_list, self.i)
    # todo ic=Tensor("GatherV2:0", shape=(?,), dtype=int64)
    # print(f'ic={ic}')
    i_emb = tf.concat(values = [
        tf.nn.embedding_lookup(item_emb_w, self.i),
        tf.nn.embedding_lookup(cate_emb_w, ic),
        ], axis=1)
    i_b = tf.gather(item_b, self.i)

    jc = tf.gather(cate_list, self.j)

    # todo 这里是什么意思？
    j_emb = tf.concat([
        tf.nn.embedding_lookup(item_emb_w, self.j),
        tf.nn.embedding_lookup(cate_emb_w, jc),
        ], axis=1)
    j_b = tf.gather(item_b, self.j)

    # 用户历史行为的embedding
    hc = tf.gather(cate_list, self.hist_i)
    h_emb = tf.concat([
        tf.nn.embedding_lookup(item_emb_w, self.hist_i),
        tf.nn.embedding_lookup(cate_emb_w, hc),
        ], axis=2)
    # todo rain--tmp_i=(<class 'numpy.ndarray'>, (32,)); hc=(32, 40); h_emb=(32, 40, 128);
    self.hc = hc
    self.h_emb = h_emb
    self.i_emb = i_emb

    # todo i_emb=Tensor("concat:0", shape=(?, 128), dtype=float32); h_emb=Tensor("concat_2:0", shape=(?, ?, 128), dtype=float32)
    # print(f'i_emb={i_emb}; h_emb={h_emb}')
    # todo hist_i 的size是 [B, 1, H]
    hist_i =attention(i_emb, h_emb, self.sl)
    #-- attention end ---

    # todo hist_i=Tensor("batch_normalization/batchnorm/add_1:0", shape=(?, 1, 128), dtype=float32)
    hist_i = tf.layers.batch_normalization(inputs = hist_i)
    # print(f'hist_i={hist_i}')
    hist_i = tf.reshape(hist_i, [-1, hidden_units], name='hist_bn')
    # todo 为什么还要做一下全连接？
    hist_i = tf.layers.dense(hist_i, hidden_units, name='hist_fcn')

    u_emb_i = hist_i
    
    hist_j =attention(j_emb, h_emb, self.sl)
    #-- attention end ---
    
    # hist_j = tf.layers.batch_normalization(inputs = hist_j)
    hist_j = tf.layers.batch_normalization(inputs = hist_j, reuse=True)
    hist_j = tf.reshape(hist_j, [-1, hidden_units], name='hist_bn')
    hist_j = tf.layers.dense(hist_j, hidden_units, name='hist_fcn', reuse=True)

    u_emb_j = hist_j
    # todo u_emb_i [None, 128]
    #  u_emb_j [None, 128]
    #  i_emb [None, 128]
    #  j_emb [None, 128]
    #  [None, 100, 128]
    print('u_emb_i',u_emb_i.get_shape().as_list())
    print('u_emb_j',u_emb_j.get_shape().as_list())
    print('i_emb',i_emb.get_shape().as_list())
    print('j_emb',j_emb.get_shape().as_list())
    #-- fcn begin -------
    # todo i_emb 是 candidate 的embedding；
    #  为什么还要将 u_emb_i * i_emb 他俩相乘呢？在 attention 的时候，不是已经将 outputs, keys 相乘了吗？
    #  难道之前得到的只是权重吗？attention 得到的权重到底是那个呢？
    #  u_emb_i 应该不是权重吧，总不能权重直接和 candidate 的embedding一起concat吧。那 attention 中的最后的output就是应该是权重
    din_i = tf.concat([u_emb_i, i_emb, u_emb_i * i_emb], axis=-1)
    din_i = tf.layers.batch_normalization(inputs=din_i, name='b1')
    d_layer_1_i = tf.layers.dense(din_i, 80, activation=tf.nn.sigmoid, name='f1')
    #if u want try dice change sigmoid to None and add dice layer like following two lines. You can also find model_dice.py in this folder.
    # d_layer_1_i = tf.layers.dense(din_i, 80, activation=None, name='f1')
    # d_layer_1_i = dice(d_layer_1_i, name='dice_1_i')
    d_layer_2_i = tf.layers.dense(d_layer_1_i, 40, activation=tf.nn.sigmoid, name='f2')
    # d_layer_2_i = tf.layers.dense(d_layer_1_i, 40, activation=None, name='f2')
    # d_layer_2_i = dice(d_layer_2_i, name='dice_2_i')
    d_layer_3_i = tf.layers.dense(d_layer_2_i, 1, activation=None, name='f3')

    din_j = tf.concat([u_emb_j, j_emb, u_emb_j * j_emb], axis=-1)
    din_j = tf.layers.batch_normalization(inputs=din_j, name='b1', reuse=True)
    d_layer_1_j = tf.layers.dense(din_j, 80, activation=tf.nn.sigmoid, name='f1', reuse=True)
    # d_layer_1_j = tf.layers.dense(din_j, 80, activation=None, name='f1', reuse=True)
    # d_layer_1_j = dice(d_layer_1_j, name='dice_1_j')
    d_layer_2_j = tf.layers.dense(d_layer_1_j, 40, activation=tf.nn.sigmoid, name='f2', reuse=True)
    # d_layer_2_j = tf.layers.dense(d_layer_1_j, 40, activation=None, name='f2', reuse=True)
    # d_layer_2_j = dice(d_layer_2_j, name='dice_2_j')
    d_layer_3_j = tf.layers.dense(d_layer_2_j, 1, activation=None, name='f3', reuse=True)
    d_layer_3_i = tf.reshape(d_layer_3_i, [-1])
    d_layer_3_j = tf.reshape(d_layer_3_j, [-1])
    x = i_b - j_b + d_layer_3_i - d_layer_3_j # [B]
    self.logits = i_b + d_layer_3_i
    
    # prediciton for selected items
    # logits for selected item:
    item_emb_all = tf.concat([
        item_emb_w,
        tf.nn.embedding_lookup(cate_emb_w, cate_list)
        ], axis=1)
    # todo 所有 item 的embedding；但是为什么这里又截取了前100个呢？每个batch都用相同的前 100 个对吧？这里预测出来是什么呢
    item_emb_sub = item_emb_all[:predict_ads_num,:]
    # todo item_emb_sub=Tensor("strided_slice_8:0", shape=(100, 128), dtype=float32);
    #  item_emb_all=Tensor("concat_7:0", shape=(63001, 128), dtype=float32)
    # print(f'item_emb_sub={item_emb_sub}; item_emb_all={item_emb_all}')
    item_emb_sub = tf.expand_dims(item_emb_sub, 0)
    # todo 这是什么意思？是取了100个样本重复了32次吗？？？
    item_emb_sub = tf.tile(item_emb_sub, [predict_batch_size, 1, 1])
    # todo item_emb_sub=Tensor("Tile_2:0", shape=(32, 100, 128), dtype=float32)
    # print(f'item_emb_sub={item_emb_sub}')
    hist_sub =attention_multi_items(item_emb_sub, h_emb, self.sl)
    #-- attention end ---
    
    hist_sub = tf.layers.batch_normalization(inputs = hist_sub, name='hist_bn', reuse=tf.AUTO_REUSE)
    # print hist_sub.get_shape().as_list() 
    hist_sub = tf.reshape(hist_sub, [-1, hidden_units])
    hist_sub = tf.layers.dense(hist_sub, hidden_units, name='hist_fcn', reuse=tf.AUTO_REUSE)

    u_emb_sub = hist_sub
    item_emb_sub = tf.reshape(item_emb_sub, [-1, hidden_units])
    din_sub = tf.concat([u_emb_sub, item_emb_sub, u_emb_sub * item_emb_sub], axis=-1)
    din_sub = tf.layers.batch_normalization(inputs=din_sub, name='b1', reuse=True)
    d_layer_1_sub = tf.layers.dense(din_sub, 80, activation=tf.nn.sigmoid, name='f1', reuse=True)
    #d_layer_1_sub = dice(d_layer_1_sub, name='dice_1_sub')
    d_layer_2_sub = tf.layers.dense(d_layer_1_sub, 40, activation=tf.nn.sigmoid, name='f2', reuse=True)
    #d_layer_2_sub = dice(d_layer_2_sub, name='dice_2_sub')
    d_layer_3_sub = tf.layers.dense(d_layer_2_sub, 1, activation=None, name='f3', reuse=True)
    d_layer_3_sub = tf.reshape(d_layer_3_sub, [-1, predict_ads_num])
    self.logits_sub = tf.sigmoid(item_b[:predict_ads_num] + d_layer_3_sub)
    self.logits_sub = tf.reshape(self.logits_sub, [-1, predict_ads_num, 1])
    #-- fcn end -------

    
    self.mf_auc = tf.reduce_mean(tf.to_float(x > 0))
    self.score_i = tf.sigmoid(i_b + d_layer_3_i)
    self.score_j = tf.sigmoid(j_b + d_layer_3_j)
    self.score_i = tf.reshape(self.score_i, [-1, 1])
    self.score_j = tf.reshape(self.score_j, [-1, 1])
    self.p_and_n = tf.concat([self.score_i, self.score_j], axis=-1)
    print('p_and_n', self.p_and_n.get_shape().as_list())


    # Step variable
    self.global_step = tf.Variable(0, trainable=False, name='global_step')
    self.global_epoch_step = \
        tf.Variable(0, trainable=False, name='global_epoch_step')
    self.global_epoch_step_op = \
        tf.assign(self.global_epoch_step, self.global_epoch_step+1)

    self.loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
            logits=self.logits,
            labels=self.y)
        )

    trainable_params = tf.trainable_variables()
    self.opt = tf.train.GradientDescentOptimizer(learning_rate=self.lr)
    gradients = tf.gradients(self.loss, trainable_params)
    clip_gradients, _ = tf.clip_by_global_norm(gradients, 5)
    self.train_op = self.opt.apply_gradients(
        zip(clip_gradients, trainable_params), global_step=self.global_step)


  def train(self, sess, uij, l):
    # todo train--tmp_i=[array([15104, 48501, 18217, 31424, 21121,  1455, 22949, 14292, 18969,
    #        12228, 32085, 41749, 24686, 41758, 36216, 51350, 46648,  8289,
    #        31145, 28432, 46749, 31995, 46186, 20975, 44581, 26003, 44262,
    #        47151, 49208, 40766, 20753, 33472], dtype=int32)]
    # hist_i,hc,h_emb,i_emb = sess.run([self.hist_i,self.hc,self.h_emb,self.i_emb], feed_dict={
    #     self.u: uij[0],
    #     self.i: uij[1],
    #     self.y: uij[2],
    #     self.hist_i: uij[3],
    #     self.sl: uij[4],
    #     self.lr: l,
    #     })
    # todo max_sl=81
    #  train--tmp_i=(<class 'numpy.ndarray'>, (32, 81)); hc=(32, 81); h_emb=(32, 81, 128)
    #  max_sl=47
    #  train--tmp_i=(<class 'numpy.ndarray'>, (32, 47)); hc=(32, 47); h_emb=(32, 47, 128); i_emb=(32, 128)
    # print(f"train--tmp_i={type(hist_i),hist_i.shape}; hc={hc.shape}; h_emb={h_emb.shape}; i_emb={i_emb.shape}")

    loss, _ = sess.run([self.loss, self.train_op], feed_dict={
        self.u: uij[0],
        self.i: uij[1],
        self.y: uij[2],
        self.hist_i: uij[3],
        self.sl: uij[4],
        self.lr: l,
        })
    return loss

  def eval(self, sess, uij):
    # todo test--tmp_i=[array([42219, 58655, 45235, 29717, 54526, 14995, 22871, 51778, 41171,...,]
    # tmp_i = sess.run([self.i],feed_dict = {self.i: uij[1]})
    # print(f"test--tmp_i={tmp_i}")

    u_auc, socre_p_and_n,score_i,score_j = sess.run([self.mf_auc, self.p_and_n,self.score_i,self.score_j], feed_dict={
        self.u: uij[0],
        self.i: uij[1],
        # todo 用户是否点击第 i 个item
        self.j: uij[2],
        self.hist_i: uij[3],
        self.sl: uij[4],
        })
    # todo socre_p_and_n=(array([[0.21905962, 0.21894753],
    #  [0.21928   , 0.2184363 ],
    #  [0.21784115, 0.21821749],
    #  ...,
    #  [0.21832946, 0.21943754],
    #  [0.21812028, 0.21824709],
    #  [0.2191213 , 0.2183209 ]], dtype=float32), 512);
    #  score_i=(array([[0.21905962],
    #  [0.21928   ],
    #  [0.21784115],
    #  [0.2191213 ]], dtype=float32), 512),
    #  score_j=(array([[0.21894753],
    #  [0.2184363 ],
    #  [0.21821749],
    #  [0.21922651],
    #  [0.21922085],
    #  [0.21842122],
    #  [0.21905398],...], dtype=float32), 512)
    # print(f"socre_p_and_n={socre_p_and_n,len(socre_p_and_n)}; score_i={score_i,len(score_i)},score_j={score_j,len(score_j)}")
    return u_auc, socre_p_and_n
  
  def test(self, sess, uij):
    return sess.run(self.logits_sub, feed_dict={
        self.u: uij[0],
        self.i: uij[1],
        self.j: uij[2],
        self.hist_i: uij[3],
        self.sl: uij[4],
        })
  

  def save(self, sess, path):
    saver = tf.train.Saver()
    saver.save(sess, save_path=path)

  def restore(self, sess, path):
    saver = tf.train.Saver()
    saver.restore(sess, save_path=path)

def extract_axis_1(data, ind):
  batch_range = tf.range(tf.shape(data)[0])
  indices = tf.stack([batch_range, ind], axis=1)
  res = tf.gather_nd(data, indices)
  return res

def attention(queries, keys, keys_length):
  '''
    H 是 embedding 的长度
    T 当前 batch 中用户历史行为的序列的最大长度
    queries 是第 i 个预测 item 的embedding
    keys：是用户点击的前 i-1 个item 的 embedding
    queries:     [B, H]
    keys:        [B, T, H]
    keys_length: [B]
  '''
  # todo keys=Tensor("concat_2:0", shape=(?, ?, 128), dtype=float32)
  # print(f'keys={keys}')
  queries_hidden_units = queries.get_shape().as_list()[-1]
  # todo queries1=Tensor("concat:0", shape=(?, 128), dtype=float32);keys=Tensor("concat_2:0", shape=(?, ?, 128), dtype=float32)
  # print(f'queries1={queries};keys={keys}')
  # todo tf.tile 同一维度上复制 [1, tf.shape(keys)[1]] 次; 是为了后面可以和 key concat
  queries = tf.tile(queries, [1, tf.shape(keys)[1]])
  # todo queries=Tensor("Tile:0", shape=(?, ?), dtype=float32)
  # print(f'queries={queries}')
  queries = tf.reshape(queries, [-1, tf.shape(keys)[1], queries_hidden_units])
  # todo  queries2=Tensor("Reshape:0", shape=(?, ?, 128), dtype=float32)
  # print(f'queries2={queries}')
  din_all = tf.concat([queries, keys, queries-keys, queries*keys], axis=-1)
  # todo din_all=Tensor("concat_3:0", shape=(?, ?, 512), dtype=float32)
  # print(f'din_all={din_all}')
  # todo tf.layers.dense(inputs,units)
  #  inputs：输入该网络层的数据
  #  units：输出的维度大小，改变inputs的最后一维
  print(f'tf.AUTO_REUSE={tf.AUTO_REUSE}')
  # todo 这里使用 dense
  d_layer_1_all = tf.layers.dense(din_all, 80, activation=tf.nn.sigmoid, name='f1_att', reuse=tf.AUTO_REUSE)
  # todo d_layer_1_all=Tensor("f1_att_1/Sigmoid:0", shape=(?, ?, 80), dtype=float32)
  # print(f'd_layer_1_all={d_layer_1_all}')
  d_layer_2_all = tf.layers.dense(d_layer_1_all, 40, activation=tf.nn.sigmoid, name='f2_att', reuse=tf.AUTO_REUSE)
  d_layer_3_all = tf.layers.dense(d_layer_2_all, 1, activation=None, name='f3_att', reuse=tf.AUTO_REUSE)
  d_layer_3_all = tf.reshape(d_layer_3_all, [-1, 1, tf.shape(keys)[1]])
  outputs = d_layer_3_all
  # Mask
  #  由于之前对所有的用户行为序列都用0扩展成当前 batch 中最长的序列长，现在需要返回每个样本中用户真是有过行为的 item；
  #  keys_length 代表每个样本中用户真实行为过的 item 长度，代表当前 batch 最大长度
  key_masks = tf.sequence_mask(keys_length, tf.shape(keys)[1])   # [B, T]
  key_masks = tf.expand_dims(key_masks, 1) # [B, 1, T]
  # todo -2 ** 32 + 1 = -4294967295
  #  为什么要用这么小的数 padding？
  paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
  outputs = tf.where(key_masks, outputs, paddings)  # [B, 1, T]

  # Scale
  #  为什么用这种方式规范化
  outputs = outputs / (keys.get_shape().as_list()[-1] ** 0.5)

  # Activation
  outputs = tf.nn.softmax(outputs)  # [B, 1, T]

  # Weighted sum
  # todo outputs=Tensor("Softmax_1:0", shape=(?, 1, ?), dtype=float32);
  #  keys=Tensor("concat_2:0", shape=(?, ?, 128), dtype=float32) 应该是 [B * T * H]; 难道是第一个维度相等的情况下，第2，3维做矩阵乘法吗？
  # print(f'outputs={outputs}; keys={keys}')
  outputs = tf.matmul(outputs, keys)  # [B, 1, H]

  return outputs

def attention_multi_items(queries, keys, keys_length):
  '''
    queries:     [B, N, H] N is the number of ads
    keys:        [B, T, H] 
    keys_length: [B]
  '''
  queries_hidden_units = queries.get_shape().as_list()[-1]  # H
  queries_nums = queries.get_shape().as_list()[1]           # N
  queries = tf.tile(queries, [1, 1, tf.shape(keys)[1]])     # 还要再重复 T 次？
  queries = tf.reshape(queries, [-1, queries_nums, tf.shape(keys)[1], queries_hidden_units]) # shape : [B, N, T, H]
  max_len = tf.shape(keys)[1]
  keys = tf.tile(keys, [1, queries_nums, 1])
  keys = tf.reshape(keys, [-1, queries_nums, max_len, queries_hidden_units]) # shape : [B, N, T, H]
  din_all = tf.concat([queries, keys, queries-keys, queries*keys], axis=-1)
  d_layer_1_all = tf.layers.dense(din_all, 80, activation=tf.nn.sigmoid, name='f1_att', reuse=tf.AUTO_REUSE)
  d_layer_2_all = tf.layers.dense(d_layer_1_all, 40, activation=tf.nn.sigmoid, name='f2_att', reuse=tf.AUTO_REUSE)
  d_layer_3_all = tf.layers.dense(d_layer_2_all, 1, activation=None, name='f3_att', reuse=tf.AUTO_REUSE)
  d_layer_3_all = tf.reshape(d_layer_3_all, [-1, queries_nums, 1, max_len])
  outputs = d_layer_3_all 
  # Mask
  key_masks = tf.sequence_mask(keys_length, max_len)   # [B, T]
  key_masks = tf.tile(key_masks, [1, queries_nums])
  key_masks = tf.reshape(key_masks, [-1, queries_nums, 1, max_len]) # shape : [B, N, 1, T]
  paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
  outputs = tf.where(key_masks, outputs, paddings)  # [B, N, 1, T]

  # Scale
  outputs = outputs / (keys.get_shape().as_list()[-1] ** 0.5)

  # Activation
  outputs = tf.nn.softmax(outputs)  # [B, N, 1, T]
  outputs = tf.reshape(outputs, [-1, 1, max_len])
  keys = tf.reshape(keys, [-1, max_len, queries_hidden_units])
  #print outputs.get_shape().as_list()
  #print keys.get_sahpe().as_list()
  # Weighted sum
  outputs = tf.matmul(outputs, keys)
  outputs = tf.reshape(outputs, [-1, queries_nums, queries_hidden_units])  # [B, N, 1, H]
  print( outputs.get_shape().as_list())
  return outputs

import numpy as np

class DataInput:
  def __init__(self, data, batch_size):

    self.batch_size = batch_size
    self.data = data
    self.epoch_size = len(self.data) // self.batch_size
    if self.epoch_size * self.batch_size < len(self.data):
      self.epoch_size += 1
    self.i = 0
    print(f"self.batch_size={self.batch_size};self.epoch_size={self.epoch_size}")

  def __iter__(self):
    return self

  def __next__(self):

    if self.i == self.epoch_size:
      raise StopIteration

    ts = self.data[self.i * self.batch_size : min((self.i+1) * self.batch_size,
                                                  len(self.data))]
    self.i += 1

    u, i, y, sl = [], [], [], []
    # todo ts=[(3112, [2493, 5180, 5912, 4951, 2500, 5258, 8896, 18028, 21920, 13164, 17814, 27575], 38958, 0),
    #  (18085, [27931, 49591, 52222], 60426, 0),
    #  (127909, [1258, 1600, 6143, 5882, 15436, 18028, 21174, 23850, 27434, 23587, 14696, 27433, 14039, 33113, 4904, 28850], 35648, 1),
    #  <class 'list'>, 32, 2608764
    # print(f'ts={ts},{type(self.data),len(ts),len(self.data)}')
    for t in ts:
      u.append(t[0])
      i.append(t[2])
      y.append(t[3])
      sl.append(len(t[1]))
    max_sl = max(sl)
    # todo max_sl=81
    #  train--tmp_i=(<class 'numpy.ndarray'>, (32, 81)); hc=(32, 81); h_emb=(32, 81, 128)
    #  max_sl=47
    #  train--tmp_i=(<class 'numpy.ndarray'>, (32, 47)); hc=(32, 47); h_emb=(32, 47, 128)
    # print(f'max_sl={max_sl}')

    hist_i = np.zeros([len(ts), max_sl], np.int64)

    k = 0
    for t in ts:
      for l in range(len(t[1])):
        hist_i[k][l] = t[1][l]
      k += 1
    # todo
    #  u:一个训练 batch 样本中用户ID集合；
    #  i:这里是根据历史上用户点击的前 i-1 个 item 预测用户对i个商品是否点击，i 就是训练 batch 样本的第 i 个 item 的集合，如果点击 y 为1，否则 y 为 0；
    #  y:用户是否点击 item i
    #  hist_i:每一个行是用户点击的前 i-1 个 item；
    #  sl：每一个数值是一个样本训练集的长度
    return self.i, (u, i, y, hist_i, sl)

class DataInputTest:
  def __init__(self, data, batch_size):

    self.batch_size = batch_size
    self.data = data
    self.epoch_size = len(self.data) // self.batch_size
    if self.epoch_size * self.batch_size < len(self.data):
      self.epoch_size += 1
    self.i = 0

  def __iter__(self):
    return self

  def __next__(self):

    if self.i == self.epoch_size:
      raise StopIteration

    ts = self.data[self.i * self.batch_size : min((self.i+1) * self.batch_size,
                                                  len(self.data))]
    self.i += 1

    u, i, j, sl = [], [], [], []
    for t in ts:
      u.append(t[0])
      i.append(t[2][0])
      j.append(t[2][1])
      sl.append(len(t[1]))
    max_sl = max(sl)

    hist_i = np.zeros([len(ts), max_sl], np.int64)

    k = 0
    for t in ts:
      for l in range(len(t[1])):
        hist_i[k][l] = t[1][l]
      k += 1
    # todo 这里使用
    #  u=([125814, 84299, 31839], 512);
    #  i=([42426, 30999, 60433], 512);
    #  为什么这里，j 是这种形式？在训练集上是 y 是否点击，测试集上对应的什么呢？
    #  j=([43514, 50490, 33271], 512);
    #  hist_i=(array([[11629,  5828, 20812, 25637,     0,     0,     0,     0,     0, 0,     0,     0,     0,     0,     0,     0,     0,     0, 0,     0,     0,     0,     0,     0,     0, ...],
    #  ...]), 512);
    #  sl=([4, 5, 5], 512)
    # print(f"u={u[:3],len(u)}; i={i[:3],len(i)}; j={j[:3],len(j)}; hist_i={hist_i[:3],len(hist_i)}; sl={sl[:3],len(sl)}")

    return self.i, (u, i, j, hist_i, sl)

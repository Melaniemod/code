import os
import time
import pickle
import random
import numpy as np
import tensorflow as tf
import sys
from input import DataInput, DataInputTest
from model import Model

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
random.seed(1234)
np.random.seed(1234)
tf.set_random_seed(1234)

train_batch_size = 32
test_batch_size = 512
predict_batch_size = 32
predict_users_num = 1000
predict_ads_num = 100


with open('dataset.pkl', 'rb') as f:
  train_set = pickle.load(f,encoding='iso-8859-1')
  test_set = pickle.load(f,encoding='iso-8859-1')
  cate_list = pickle.load(f,encoding='iso-8859-1')
  user_count, item_count, cate_count = pickle.load(f,encoding='iso-8859-1')

# todo  train_set=[(104760, [3737, 19450], 18486, 1), (129282, [3647, 4342, 6855, 3805], 4206, 1), (130232, [1805, 4309], 21354, 1)];
#  这里 test_set[2] 是一个 tuple 是什么意思？tuple 对应的两个元素是代表什么？
#  test_set=[(91788, [16942, 42346, 38112, 36550, 45547, 31289, 48828, 38115, 51873, 25727, 32576, 34688, 49156, 40120, 56219, 54422], (57905, 11716)), (133148, [23836, 39599, 46106, 41622], (57608, 49499)), (176022, [58837, 60672, 60967, 57243], (20192, 4966))];
#  cate_list=[738 157 571 ...  63 674 351];
#  user_count=192403;
#  item_count=63001;
#  cate_count=801;
#  len(test_set)=192403;
#  len(train_set)=2608764;
#  len(cate_list) = 63001 -- 这里 cate_list 应该是每个 item 对应的类目编码
# print(f"train_set={train_set[:3]}; test_set={test_set[:3]}; cate_list={cate_list}; user_count={user_count}; "
#       f"item_count={item_count}; cate_count={cate_count}; len(test_set)={len(test_set)};len(train_set)={len(train_set)};len(cate_list) = {len(cate_list)}")

best_auc = 0.0

def calc_auc(raw_arr):
    """Summary
    Args:
        raw_arr (TYPE): Description
    Returns:
        TYPE: Description
    """
    # sort by pred value, from small to big
    arr = sorted(raw_arr, key=lambda d:d[2])

    auc = 0.0
    fp1, tp1, fp2, tp2 = 0.0, 0.0, 0.0, 0.0
    for record in arr:
        fp2 += record[0] # noclick
        tp2 += record[1] # click
        auc += (fp2 - fp1) * (tp2 + tp1)
        fp1, tp1 = fp2, tp2

    # if all nonclick or click, disgard
    threshold = len(arr) - 1e-3
    if tp2 > threshold or fp2 > threshold:
        return -0.5

    if tp2 * fp2 > 0.0:  # normal auc
        return (1.0 - auc / (2.0 * tp2 * fp2))
    else:
        return None

def _auc_arr(score):
  # print(f'score={score}')
  score_p = score[:,0]
  score_n = score[:,1]
  #print "============== p ============="
  #print score_p
  #print "============== n ============="
  #
  #print score_n
  score_arr = []
  for s in score_p.tolist():
    score_arr.append([0, 1, s])
  for s in score_n.tolist():
    score_arr.append([1, 0, s])
  return score_arr

def _eval(sess, model):
  auc_sum = 0.0
  score_arr = []
  # print('test_set, predict_batch_size')
  for _, uij in DataInputTest(test_set, test_batch_size):
    auc_, score_ = model.eval(sess, uij)
    score_arr += _auc_arr(score_)
    auc_sum += auc_ * len(uij[0])
    # todo score_arr=1024
    #  score_arr=2048
    #  score_arr=3072
    # print(f'score_arr={len(score_arr)}')
  # 当前验证集的平均 gauc
  test_gauc = auc_sum / len(test_set)
  Auc = calc_auc(score_arr)
  global best_auc
  if best_auc < test_gauc:
    best_auc = test_gauc
    model.save(sess, 'save_path/ckpt')
  return test_gauc, Auc

def _test(sess, model):
  auc_sum = 0.0
  score_arr = []
  predicted_users_num = 0
  print("test sub items")
  for _, uij in DataInputTest(test_set, predict_batch_size):
    if predicted_users_num >= predict_users_num:
        break
    score_ = model.test(sess, uij)
    score_arr.append(score_)
    predicted_users_num += predict_batch_size
  return score_[0]

gpu_options = tf.GPUOptions(allow_growth=True)
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:

  print(user_count, item_count, cate_count, cate_list, predict_batch_size, predict_ads_num)

  model = Model(user_count, item_count, cate_count, cate_list, predict_batch_size, predict_ads_num)
  sess.run(tf.global_variables_initializer())
  sess.run(tf.local_variables_initializer())

  print('test_gauc: %.4f\t test_auc: %.4f' % _eval(sess, model))
  sys.stdout.flush()
  lr = 1.0
  start_time = time.time()
  for _ in range(1):

    random.shuffle(train_set)

    epoch_size = round(len(train_set) / train_batch_size)
    loss_sum = 0.0

    for _, uij in DataInput(train_set, train_batch_size):
      # todo
      #  u:一个训练 batch 样本中用户ID集合；
      #  i:这里是根据历史上用户点击的前 i-1 个 item 预测用户对i个商品是否点击，i 就是训练 batch 样本的第 i 个 item 的集合，如果点击 y 为1，否则 y 为 0；
      #  y  用户是否点击 item i
      #  hist_i:每一个行是用户点击的前 i-1 个 item；
      #  sl：每一个数值是一个样本训练集的长度
      #  uij ([187945, 171122, 54012, 105794, 41087, 119660, 174921, 110025, 19630, 60142, 47039, 138134, 20182, 49234, 153674, 122133, 86479, 187121, 158649, 19724, 126080, 182226, 85078, 139526, 62744, 22586, 129030, 149921, 178551, 129594, 175383, 52738],
      #  [5023, 24332, 11407, 59208, 39085, 30706, 48270, 29314, 7876, 39323, 684, 47915, 18414, 43676, 21320, 6851, 53172, 60933, 28232, 44496, 16150, 58216, 24309, 42401, 32965, 13617, 12485, 2860, 37136, 18767, 19251, 51515],
      #  [0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0],
      #  array([[40586, 37040, 45561, ...,     0,     0,     0],
      #         [10291, 20811, 18650, ...,     0,     0,     0],
      #         [ 6726, 10322, 21439, ...,     0,     0,     0],
      #         ...,
      #         [ 3492, 24661, 28379, ...,     0,     0,     0],
      #         [25564, 26291, 33442, ...,     0,     0,     0],
      #         [62294, 14486,     0, ...,     0,     0,     0]]),
      #  [3, 21, 3, 3, 2, 2, 7, 42, 5, 3, 7, 4, 1, 1, 2, 7, 2, 15, 4, 2, 17, 4, 5, 15, 23, 3, 18, 18, 4, 3, 18, 2])
      # print(f'uij={uij}; type(uij)={type(uij)}; ')
      loss = model.train(sess, uij, lr)
      loss_sum += loss

      if model.global_step.eval() % 1000 == 0:
        test_gauc, Auc = _eval(sess, model)
        print('Epoch %d Global_step %d\tTrain_loss: %.4f\tEval_GAUC: %.4f\tEval_AUC: %.4f' %
              (model.global_epoch_step.eval(), model.global_step.eval(),
               loss_sum / 1000, test_gauc, Auc))
        sys.stdout.flush()
        loss_sum = 0.0

      if model.global_step.eval() % 336000 == 0:
        lr = 0.1

    print('Epoch %d DONE\tCost time: %.2f' %
          (model.global_epoch_step.eval(), time.time()-start_time))
    sys.stdout.flush()
    model.global_epoch_step_op.eval()

  print('best test_gauc:', best_auc)
  sys.stdout.flush()

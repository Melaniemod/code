import random
import pickle

random.seed(1234)

with open('../raw_data/remap.pkl', 'rb') as f:
  reviews_df = pickle.load(f)
  cate_list = pickle.load(f)
  user_count, item_count, cate_count, example_count = pickle.load(f)

train_set = []
test_set = []
# todo reviews_df: <class 'pandas.core.frame.DataFrame'>
#  <class 'numpy.ndarray'> <class 'int'> <class 'int'> <class 'int'> <class 'int'>
# print('reviews_df:',type(reviews_df),type(cate_list),type(user_count),type(item_count),type(cate_count),type(example_count))
# todo reviewerID   asin  unixReviewTime
# 	0           0  13179      1400457600
# 	1           0  17993      1400457600
# 	2           0  28326      1400457600
# 	3           0  29247      1400457600
# 	4           0  62275      1400457600
# print(reviews_df.head())
# todo <pandas.core.groupby.generic.DataFrameGroupBy object at 0x7fa22cf8e128>
# print(reviews_df.groupby('reviewerID'))

# todo cate_list
# print(f'cate_list={cate_list}')

for reviewerID, hist in reviews_df.groupby('reviewerID'):
  pos_list = hist['asin'].tolist()
  # todo reviewerID=98,
  #  hist= reviewerID   asin  unixReviewTime
  #  650          98  53178      1357862400
  #  651          98  51349      1358208000
  #  652          98  50557      1360022400
  #  653          98  26301      1363824000
  #  654          98  32991      1380153600
  #  655          98  57362      1394323200,
  #   pos_list=[53178, 51349, 50557, 26301, 32991, 57362]
  # print(f"reviewerID={reviewerID}, \n hist={hist}, \n pos_list={pos_list}")
  def gen_neg():
    neg = pos_list[0]
    while neg in pos_list:
      # todo 这里是要生成一个用户没有点击的负样本
      neg = random.randint(0, item_count-1)
    return neg
  neg_list = [gen_neg() for i in range(len(pos_list))]

  # todo 训练集是前 i-1 个样本，label 第i个样本
  #  最后一次看的电影作为测试集的样本
  for i in range(1, len(pos_list)):
    hist = pos_list[:i]
    if i != len(pos_list) - 1:
      train_set.append((reviewerID, hist, pos_list[i], 1))
      train_set.append((reviewerID, hist, neg_list[i], 0))
    else:
      label = (pos_list[i], neg_list[i])
      test_set.append((reviewerID, hist, label))

random.shuffle(train_set)
random.shuffle(test_set)

assert len(test_set) == user_count
# assert(len(test_set) + len(train_set) // 2 == reviews_df.shape[0])

with open('dataset.pkl', 'wb') as f:
  pickle.dump(train_set, f, pickle.HIGHEST_PROTOCOL)
  pickle.dump(test_set, f, pickle.HIGHEST_PROTOCOL)
  pickle.dump(cate_list, f, pickle.HIGHEST_PROTOCOL)
  pickle.dump((user_count, item_count, cate_count), f, pickle.HIGHEST_PROTOCOL)



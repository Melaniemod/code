# -*-coding:utf-8-*-



import utils

train_file = '../data/train.txt'
test_file = '../data/test.txt'

input_dim = utils.INPUT_DIM

train_data = utils.read_data(train_file)
# train_data = pkl.load(open('../data/train.yx.pkl', 'rb'))
train_data = utils.shuffle(train_data)
test_data = utils.read_data(test_file)

print(type(train_data),train_data)

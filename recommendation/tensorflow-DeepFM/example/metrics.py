
import numpy as np

# todo 两个gini结果应该是一样的，计算的时候上面这个按照预测值从大到小排的，下面这个按照从小到大排的，两个函数在-giniSum的时候不一样就可以吧？
#  代码中gini-Normalization 可能和博主楼上解释的稍微难理解些。代码中是按照预测值倒叙排序的（all = all[np.lexsort((all[:, 2], -1 * all[:, 1]))]），后面减面积的时候”giniSum -= (len(actual) + 1) / 2.“，如果all = all[np.lexsort((all[:, 2], 1 * all[:, 1]))]以及giniSum = (len(actual) + 1) / 2. -giniSum，就和楼主解释的一摸一样了。
def gini(actual, pred):
    assert (len(actual) == len(pred))
    # todo np.r_是按列连接两个矩阵，就是把两矩阵上下相加，要求列数相等，类似于pandas中的concat()。
    #  np.c_是按行连接两个矩阵，就是把两矩阵左右相加，要求行数相等，类似于pandas中的merge()。
    all = np.asarray(np.c_[actual, pred, np.arange(len(actual))], dtype=np.float)
    # todo np.lexsort((b,a)) # Sort by a, then by b。先按照a列表排序，对于a中相同的元素，按照列表b排序
    all = all[np.lexsort((all[:, 2], -1 * all[:, 1]))]
    # todo
    # print(f"all:{all}")
    totalLosses = all[:, 0].sum()
    # todo 除以totalLosses是将这个数规划到 0～1之间
    giniSum = all[:, 0].cumsum().sum() / totalLosses
    # print(f"all[:, 0].cumsum().sum()={all[:, 0].cumsum().sum()}；all[:, 0].cumsum()={all[:, 0].cumsum()};giniSum={giniSum}；totalLosses={totalLosses}；len(actual)={len(actual)}")

    giniSum -= (len(actual) + 1) / 2.
    # print(f"giniSum={giniSum}; giniSum / len(actual)={giniSum / len(actual)}")
    return giniSum / len(actual)


def gini(actual, pred):
    assert (len(actual) == len(pred))
    # todo np.r_是按列连接两个矩阵，就是把两矩阵上下相加，要求列数相等，类似于pandas中的concat()。
    #  np.c_是按行连接两个矩阵，就是把两矩阵左右相加，要求行数相等，类似于pandas中的merge()。
    all = np.asarray(np.c_[actual, pred, np.arange(len(actual))], dtype=np.float)
    # todo np.lexsort((b,a)) # Sort by a, then by b。先按照a列表排序，对于a中相同的元素，按照列表b排序
    all = all[np.lexsort((all[:, 2], 1 * all[:, 1]))]
    # todo
    # print(f"all:{all}")
    totalLosses = all[:, 0].sum()
    # todo 除以totalLosses是将这个数规划到 0～1之间
    giniSum = all[:, 0].cumsum().sum() / totalLosses
    # print(f"all[:, 0].cumsum().sum()={all[:, 0].cumsum().sum()}；all[:, 0].cumsum()={all[:, 0].cumsum()};giniSum={giniSum}；totalLosses={totalLosses}；len(actual)={len(actual)}")

    giniSum = (len(actual) + 1) / 2. -giniSum
    # print(f"giniSum={giniSum}; giniSum / len(actual)={giniSum / len(actual)}")
    return giniSum / len(actual)

def gini_norm(actual, pred):
    # print(f"gini(actual, pred) / gini(actual, actual)={gini(actual, pred) / gini(actual, actual)}")
    return gini(actual, pred) / gini(actual, actual)

if __name__ == '__main__':
    predictions = [0.9, 0.3, 0.8, 0.75, 0.65, 0.6, 0.78, 0.7, 0.05, 0.4, 0.4, 0.05, 0.5, 0.1, 0.1]
    actual = [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    # todo all:[[ 1.    0.9   0.  ]
    #  [ 1.    0.8   2.  ]
    #  [ 0.    0.78  6.  ]
    #  [ 1.    0.75  3.  ]
    #  [ 0.    0.7   7.  ]
    #  [ 1.    0.65  4.  ]
    #  [ 1.    0.6   5.  ]
    #  [ 0.    0.5  12.  ]
    #  [ 0.    0.4   9.  ]
    #  [ 0.    0.4  10.  ]
    #  [ 1.    0.3   1.  ]
    #  [ 0.    0.1  13.  ]
    #  [ 0.    0.1  14.  ]
    #  [ 0.    0.05  8.  ]
    #  [ 0.    0.05 11.  ]]
    #  all[:, 0].cumsum().sum()=65.0；all[:, 0].cumsum()=[1. 2. 2. 3. 3. 4. 5. 5. 5. 5. 6. 6. 6. 6. 6.];giniSum=10.833333333333334；totalLosses=6.0；len(actual)=15
    #  giniSum=2.833333333333334; giniSum / len(actual)=0.18888888888888894
    #  对于真实值或者，最好的模型，cumsum() 会一步一步增加至最大值，而不是突然增加到最大值
    rec_gini=gini_norm(actual, predictions)
    print(rec_gini)




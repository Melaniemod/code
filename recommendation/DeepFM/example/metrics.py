
import numpy as np

def gini(actual, pred):
    assert (len(actual) == len(pred))
    # todo np.r_是按列连接两个矩阵，就是把两矩阵上下相加，要求列数相等，类似于pandas中的concat()。
    #  np.c_是按行连接两个矩阵，就是把两矩阵左右相加，要求行数相等，类似于pandas中的merge()。
    all = np.asarray(np.c_[actual, pred, np.arange(len(actual))], dtype=np.float)
    # todo np.lexsort((b,a)) # Sort by a, then by b。先按照a列表排序，对于a中相同的元素，按照列表b排序
    # todo all=[[0.00000000e+00 1.06796145e-01 0.00000000e+00]
    #  [0.00000000e+00 6.58136904e-02 1.00000000e+00]
    #  [0.00000000e+00 7.64078498e-02 2.00000000e+00]
    #  ...
    #  [0.00000000e+00 9.03150141e-02 3.96805000e+05]
    #  [0.00000000e+00 6.97593689e-02 3.96806000e+05]
    #  [0.00000000e+00 8.36791396e-02 3.96807000e+05]]
    # print(f"all={all}")
    all = all[np.lexsort((all[:, 2], -1 * all[:, 1]))]
    # todo all:[[0.00000000e+00 1.88046962e-01 2.97560000e+04]
    #  [0.00000000e+00 1.83427185e-01 2.35617000e+05]
    #  [0.00000000e+00 1.79942161e-01 7.07420000e+04]
    #  ...
    #  [0.00000000e+00 3.40999961e-02 1.20222000e+05]
    #  [0.00000000e+00 3.39312255e-02 2.72300000e+03]
    #  [0.00000000e+00 3.31828296e-02 2.21453000e+05]]
    # print(f"all:{all}")
    totalLosses = all[:, 0].sum()
    giniSum = all[:, 0].cumsum().sum() / totalLosses

    giniSum -= (len(actual) + 1) / 2.
    return giniSum / len(actual)

def gini_norm(actual, pred):
    # todo gini(actual, actual)=0.4817758210519949
    #  gini(actual, actual)=0.4817770811072357
    # print(f"gini(actual, actual)={gini(actual, actual)}")
    return gini(actual, pred) / gini(actual, actual)

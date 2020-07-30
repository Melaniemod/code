"""
A data parser for Porto Seguro's Safe Driver Prediction competition's dataset.
URL: https://www.kaggle.com/c/porto-seguro-safe-driver-prediction
"""
import pandas as pd


class FeatureDictionary(object):
    def __init__(self, trainfile=None, testfile=None,
                 dfTrain=None, dfTest=None, numeric_cols=[], ignore_cols=[]):
        assert not ((trainfile is None) and (dfTrain is None)), "trainfile or dfTrain at least one is set"
        assert not ((trainfile is not None) and (dfTrain is not None)), "only one can be set"
        assert not ((testfile is None) and (dfTest is None)), "testfile or dfTest at least one is set"
        assert not ((testfile is not None) and (dfTest is not None)), "only one can be set"
        self.trainfile = trainfile
        self.testfile = testfile
        self.dfTrain = dfTrain
        self.dfTest = dfTest
        self.numeric_cols = numeric_cols
        self.ignore_cols = ignore_cols
        self.gen_feat_dict()

    def gen_feat_dict(self):
        if self.dfTrain is None:
            dfTrain = pd.read_csv(self.trainfile)
        else:
            dfTrain = self.dfTrain
        if self.dfTest is None:
            dfTest = pd.read_csv(self.testfile)
        else:
            dfTest = self.dfTest
        df = pd.concat([dfTrain, dfTest])
        self.feat_dict = {}
        # tc 特征维度
        tc = 0
        for col in df.columns:
            if col in self.ignore_cols:
                continue
            if col in self.numeric_cols:
                # map to a single index
                self.feat_dict[col] = tc
                tc += 1
            else:
                us = df[col].unique()
                self.feat_dict[col] = dict(zip(us, range(tc, len(us)+tc)))
                tc += len(us)
        # todo self.feat_dict  {'missing_feat': 0, ... 'ps_car_02_cat': {1: 14, 0: 15, -1: 16}, ...
        #  'ps_ind_18_bin': {0: 254, 1: 255}, 'ps_reg_01': 256, 'ps_reg_02': 257, 'ps_reg_03': 258}
        # print("self.feat_dict ",self.feat_dict)
        self.feat_dim = tc


class DataParser(object):
    def __init__(self, feat_dict):
        self.feat_dict = feat_dict

    def parse(self, infile=None, df=None, has_label=False):
        assert not ((infile is None) and (df is None)), "infile or df at least one is set"
        assert not ((infile is not None) and (df is not None)), "only one can be set"
        if infile is None:
            dfi = df.copy()
        else:
            dfi = pd.read_csv(infile)
        if has_label:
            y = dfi["target"].values.tolist()
            dfi.drop(["id", "target"], axis=1, inplace=True)
        else:
            ids = dfi["id"].values.tolist()
            dfi.drop(["id"], axis=1, inplace=True)
        # dfi for feature index
        # dfv for feature value which can be either binary (1/0) or float (e.g., 10.24)
        dfv = dfi.copy()
        # todo   ps_ind_01  ps_ind_02_cat  ...  missing_feat  ps_car_13_x_ps_reg_03
        # 0          2              2  ...             1               0.634544
        # 1          1              1  ...             2               0.474062
        # 2          5              4  ...             3              -0.641586
        # 3          0              1  ...             0               0.315425
        # 4          0              2  ...             2               0.475728
        # print("dfi.head()",dfi.head())
        for col in dfi.columns:
            if col in self.feat_dict.ignore_cols:
                dfi.drop(col, axis=1, inplace=True)
                dfv.drop(col, axis=1, inplace=True)
                continue
            if col in self.feat_dict.numeric_cols:
                # 如果是数值型，整列是一样的数字
                dfi[col] = self.feat_dict.feat_dict[col]
            else:
                # todo self.feat_dict.feat_dict[col] ps_ind_16_bin {0: 250, 1: 251}
                #  self.feat_dict.feat_dict[col] ps_ind_17_bin {1: 252, 0: 253}
                #  self.feat_dict.feat_dict[col] ps_ind_18_bin {0: 254, 1: 255}
                # print("self.feat_dict.feat_dict[col]",col,self.feat_dict.feat_dict[col])
                dfi[col] = dfi[col].map(self.feat_dict.feat_dict[col])
                dfv[col] = 1.
                # print("111",dfv[col].head())
                # break
        # todo    ps_ind_01  ps_ind_02_cat  ...  missing_feat  ps_car_13_x_ps_reg_03
        # 0        179            187  ...             0                    176
        # 1        180            188  ...             0                    176
        # 2        181            189  ...             0                    176
        # 3        182            188  ...             0                    176
        # 4        182            187  ...             0                    176
        # [5 rows x 39 columns]
        #       ps_ind_01  ps_ind_02_cat  ...  missing_feat  ps_car_13_x_ps_reg_03
        # 0        1.0            1.0  ...             1               0.634544
        # 1        1.0            1.0  ...             2               0.474062
        # 2        1.0            1.0  ...             3              -0.641586
        # 3        1.0            1.0  ...             0               0.315425
        # 4        1.0            1.0  ...             2               0.475728
        # [5 rows x 39 columns] (595212, 39) (595212, 39)
        # print(dfi.head(),dfv.head(),dfi.shape,dfv.shape)
        # list of list of feature indices of each sample in the dataset
        # print(f"dfi['ps_car_13_x_ps_reg_03']={dfi['ps_car_13_x_ps_reg_03'].head()}")
        Xi = dfi.values.tolist()
        # list of list of feature values of each sample in the dataset
        Xv = dfv.values.tolist()
        # todo Xi=[179, 187, 192, 204, 207, 215, 217, 219, 221, 223, 225, 227, 229, 231, 236, 250, 252, 254, 256, 257, 258, 1, 14, 17, 20, 30, 33, 51, 54, 56, 62, 70, 65, 174, 175, 177, 178, 0, 176];
        #  Xv=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.7, 0.2, 0.7180703307999999, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.4, 0.8836789178, 0.3708099244, 3.6055512755, 1.0, 0.6345436128256319];
        #  595212,39;
        # print(f"Xi={Xi[0]}; Xv={Xv[0]}; {len(Xi)},{len(Xi[0])};")
        if has_label:
            return Xi, Xv, y
        else:
            return Xi, Xv, ids


# -*-coding:utf-8-*-
import pandas as pd
import numpy as np


import config
from DataReader import FeatureDictionary, DataParser


def _load_data():

    dfTrain = pd.read_csv(config.TRAIN_FILE)
    dfTest = pd.read_csv(config.TEST_FILE)

    def preprocess(df):
        cols = [c for c in df.columns if c not in ["id", "target"]]
        df["missing_feat"] = np.sum((df[cols] == -1).values, axis=1)
        df["ps_car_13_x_ps_reg_03"] = df["ps_car_13"] * df["ps_reg_03"]
        return df

    dfTrain = preprocess(dfTrain)
    dfTest = preprocess(dfTest)

    cols = [c for c in dfTrain.columns if c not in ["id", "target"]]
    cols = [c for c in cols if (not c in config.IGNORE_COLS)]

    X_train = dfTrain[cols].values
    y_train = dfTrain["target"].values
    X_test = dfTest[cols].values
    ids_test = dfTest["id"].values
    cat_features_indices = [i for i,c in enumerate(cols) if c in config.CATEGORICAL_COLS]

    return dfTrain, dfTest, X_train, y_train, X_test, ids_test, cat_features_indices




# load data
# dfTrain, dfTest, X_train, y_train, X_test, ids_test, cat_features_indices = _load_data()
#
# fd = FeatureDictionary(dfTrain=dfTrain, dfTest=dfTest,
#                        numeric_cols=config.NUMERIC_COLS,
#                        ignore_cols=config.IGNORE_COLS)
#
# data_parser = DataParser(feat_dict=fd)
#
# # print(dfTrain.head())
#
# Xi_train, Xv_train, y_train = data_parser.parse(df=dfTrain, has_label=True)
# Xi_test, Xv_test, ids_test = data_parser.parse(df=dfTest)



a=[1,2,3,4,5,6]
b=['a','b','c','c']

c=a+b
print(c)



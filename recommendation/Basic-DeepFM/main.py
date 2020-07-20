# -*-coding:utf-8-*-

import os
import sys

import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.metrics import make_scorer
from sklearn.model_selection import StratifiedKFold

import config
from metrics import gini_norm
from DataReader import FeatureDictionary, DataParser
sys.path.append("..")
from DeepFM import DeepFM

gini_scorer = make_scorer(gini_norm, greater_is_better=True, needs_proba=True)

def _load_data():
    dfTrain = pd.read_csv(config.TRAIN_FILE)
    dfTest = pd.read_csv(config.TEST_FILE)

    def preprocess(df):
        col = [c for c in df.columns if c not in ["id","target"]]
        df['missing_feat'] = np.sum((df[col] == -1).values,axis=1)
        df["ps_car_13_x_ps_reg_03"] = df['ps_car_13'] * df['ps_reg_03']
        return df

    dfTrain = preprocess(dfTrain)
    dfTest = preprocess(dfTest)

    cols = [c for c in dfTrain.columns if c not in ["id","target"]]
    cols = [c for c in cols if (c not in config.IGNORE_COLS)]

    Xtrain = dfTrain[cols].values
    Xtest = dfTest[cols].values
    y_train = dfTrain['target'].values
    ids_test = dfTest['id'].values
    cat_features_indices = [i for i,c in enumerate(cols) if c in config.CATEGORICAL_COLS]

    return dfTrain,dfTest,Xtrain,y_train,Xtest,ids_test,cat_features_indices


def _run_base_model_dfm(dfTrain,dfTest,folds,dfm_params):

    fd = FeatureDictionary(dfTrain=dfTrain,dfTest=dfTest,numeric_cols=config.NUMERIC_COLS
                           ,ignore_cols=config.IGNORE_COLS)
    data_parser = DataParser(fd)
    Xi_train,Xv_train,y_train = data_parser.parse(df=dfTrain,has_label=True)
    Xi_test,Xv_test,ids_test = data_parser.parse(df=dfTest,has_label=False)

    dfm_params['feature_size'] = fd.feat_dim
    dfm_params['field_size'] = len(Xi_train[0])

    y_train_meta = np.zeros((dfTrain.shape[0],1),dtype=float)
    y_test_meta = np.zeros((dfTest.shape[0],1),dtype=float)

    gini_result_cv = np.zeros(len(folds),dtype=float)
    gini_result_epoch_train = np.zeros((len(folds),dfm_params['epoch']),dtype=float)
    gini_result_epoch_valid = np.zeros((len(folds),dfm_params["epoch"]),dtype=float)

    _get = lambda x,l:[x[i] for i in l]
    for i,(train_idx,valid_idx) in enumerate(folds):
        Xi_train_,Xv_train_,y_train_ = _get(Xi_train,train_idx),_get(Xv_train,train_idx),_get(y_train,train_idx)
        Xi_valid_,Xv_valid_,y_valid_ = _get(Xi_train,valid_idx),_get(Xv_train,valid_idx), _get(y_train,valid_idx)

        dfm = DeepFM(**dfm_params)
        dfm.fit(Xi_train_,Xv_train_,y_train_,Xi_valid_,Xv_valid_,y_valid_)

        y_train_meta[valid_idx,0] = dfm.predict(Xi_valid_,Xv_valid_)
        y_test_meta[:,0] += dfm.predict(Xi_test,Xv_test)

        gini_result_cv[i] = gini_norm(y_valid_,y_train_meta[valid_idx])
        gini_result_epoch_train[i]=dfm.train_result
        gini_result_epoch_valid[i] = dfm.valid_result

    y_test_meta /=float(len(folds))

    if dfm_params["use_fm"] and dfm_params['use_deep']:
        clf_str = "DeepFM"
    elif dfm_params["use_fm"]:
        clf_str = "FM"
    elif dfm_params["use_deep"]:
        clf_str = "DNN"
    file_name = "%s_Mean%.5f_Std%.5f.csv" %(clf_str,gini_result_cv.mean(),gini_result_cv.std())
    _make_submission(ids_test,y_test_meta,file_name)

    return y_train_meta,y_test_meta


def _make_submission(ids_test,y_pred,filename):
    pd.DataFrame({"id":ids_test,"predict":y_pred.flatten()}).to_csv(
        os.path.join(config.SUB_DIR,filename),index=False,float_format="%.5f"
    )


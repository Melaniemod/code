from collections import UserList, UserDict
from typing import Union, Iterable, Tuple, Dict, Any

import tensorflow as tf
import numpy as np
import pandas as pd
import pickle
import os.path


def _meta_file(path):
    return os.path.join(path, 'feeder_meta.pkl')


class VarFeeder:
    """
    Helper to avoid feed_dict and manual batching. Maybe I had to use TFRecords instead.
    Builds temporary TF graph, injects variables into, and saves variables to TF checkpoint.
    In a train time, variables can be built by build_vars() and content restored by FeederVars.restore()
    """
    def __init__(self, path: str,
                 tensor_vars: Dict[str, Union[pd.DataFrame, pd.Series, np.ndarray]] = None,
                 plain_vars: Dict[str, Any] = None):
        """
        :param path: dir to store data
        :param tensor_vars: Variables to save as Tensors (pandas DataFrames/Series or numpy arrays)
        :param plain_vars: Variables to save as Python objects
        """
        tensor_vars = tensor_vars or dict()

        def get_values(v):
            """
            todo 格式统一，对数据类型以及数值类型统一化
            """
            v = v.values if hasattr(v, 'values') else v
            if not isinstance(v, np.ndarray):
                v = np.array(v)
            if v.dtype == np.float64:
                v = v.astype(np.float32)
            return v

        values = [get_values(var) for var in tensor_vars.values()]

        # todo 这里 tensor_vars 是一个字典，取value和取name的时候不一起取，会不会对不上呢？
        self.shapes = [var.shape for var in values]
        self.dtypes = [v.dtype for v in values]
        self.names = list(tensor_vars.keys())
        self.path = path
        self.plain_vars = plain_vars
        # print("self.shapes,self.dtypes,self.names",self.shapes,self.dtypes,self.names)

        if not os.path.exists(path):
            os.mkdir(path)

        with open(_meta_file(path), mode='wb') as file:
            # todo 序列化对象，并将结果数据流写入到文件对象中。
            pickle.dump(self, file)


        with tf.Graph().as_default():
            tensor_vars = self._build_vars()
            # todo placeholder，占位符，在tensorflow中类似于函数参数，运行时必须传入值。
            placeholders = [tf.placeholder(tf.as_dtype(dtype), shape=shape) for dtype, shape in
                            zip(self.dtypes, self.shapes)]
            # todo tf.assign(A, new_number): 这个函数的功能主要是把A的值变为new_number
            #  这一波骚操作是干嘛？先把  tensor值变为 placeholder，后面又以feed_dict的形式将tensor喂进去。什么要这么干？
            #  首先，tensor_vars是没有值的只有shape和数据类型。
            #  tensor_vars -- placeholders； placeholders -- values这个节奏吗？
            assigners = [tensor_var.assign(placeholder) for tensor_var, placeholder in
                         zip(tensor_vars, placeholders)]
            feed = {ph: v for ph, v in zip(placeholders, values)}
            saver = tf.train.Saver(self._var_dict(tensor_vars), max_to_keep=1)
            init = tf.global_variables_initializer()

            with tf.Session(config=tf.ConfigProto(device_count={'GPU': 0})) as sess:
                sess.run(init)
                sess.run(assigners, feed_dict=feed)
                save_path = os.path.join(path, 'feeder.cpt')
                # todo 这样save的时候都save了那些东西？整个对象都save了吗？
                saver.save(sess, save_path, write_meta_graph=False, write_state=False)

    def _var_dict(self, variables):
        return {name: var for name, var in zip(self.names, variables)}

    def _build_vars(self):
        def make_tensor(shape, dtype, name):
            # todo 将[pd.DataFrame, pd.Series, np.ndarray]等数据类型转化为tensor。name是tensor_vars的键值
            tf_type = tf.as_dtype(dtype)
            if tf_type == tf.string:
                empty = ''
            elif tf_type == tf.bool:
                empty = False
            else:
                empty = 0
            init = tf.constant(empty, shape=shape, dtype=tf_type)
            return tf.get_local_variable(name=name, initializer=init, dtype=tf_type)

        with tf.device("/cpu:0"):
            with tf.name_scope('feeder_vars'):
                return [make_tensor(shape, dtype, name) for shape, dtype, name in
                        zip(self.shapes, self.dtypes, self.names)]

    def create_vars(self):
        """
        todo 为什么加载模型的时候，对这个对象做处理返回FeederVars的对象，而且FeederVars只是将VarFeeder的plain_vars添加到自己的属性中，为什么要这样做呢？
        Builds variable list to use in current graph. Should be called during graph building stage
        :return: variable list with additional restore and create_saver methods
        """
        return FeederVars(self._var_dict(self._build_vars()), self.plain_vars, self.path)

    @staticmethod
    def read_vars(path):
        with open(_meta_file(path), mode='rb') as file:
            feeder = pickle.load(file)
        assert feeder.path == path
        return feeder.create_vars()


class FeederVars(UserDict):
    def __init__(self, tensors: dict, plain_vars: dict, path):
        variables = dict(tensors)
        # todo variables 1 =
        #  variables 1 =
        #  {'hits': <tf.Variable 'hits:0' shape=(145036, 805) dtype=float32_ref>,
        #  'lagged_ix': <tf.Variable 'lagged_ix:0' shape=(867, 4) dtype=int16_ref>,
        #  'page_map': <tf.Variable 'page_map:0' shape=(52752, 4) dtype=int32_ref>,
        #  'page_ix': <tf.Variable 'page_ix:0' shape=(145036,) dtype=string_ref>,
        #  'pf_agent': <tf.Variable 'pf_agent:0' shape=(145036, 4) dtype=float32_ref>,
        #  'pf_country': <tf.Variable 'pf_country:0' shape=(145036, 7) dtype=float32_ref>,
        #  'pf_site': <tf.Variable 'pf_site:0' shape=(145036, 3) dtype=float32_ref>,
        #  'page_popularity': <tf.Variable 'page_popularity:0' shape=(145036,) dtype=float32_ref>,
        #  'year_autocorr': <tf.Variable 'year_autocorr:0' shape=(145036,) dtype=float32_ref>,
        #  'quarter_autocorr': <tf.Variable 'quarter_autocorr:0' shape=(145036,) dtype=float32_ref>,
        #  'dow': <tf.Variable 'dow:0' shape=(867, 2) dtype=float32_ref>};
        #  plain_vars=
        #  {'features_days': 867, 'data_days': 805, 'n_pages': 145036, 'data_start': '2015-07-01',
        #  'data_end': Timestamp('2017-09-11 00:00:00'), 'features_end': Timestamp('2017-11-13 00:00:00')}
        #  variables 2 =
        #  {'hits': <tf.Variable 'hits:0' shape=(145036, 805) dtype=float32_ref>,
        #  'lagged_ix': <tf.Variable 'lagged_ix:0' shape=(867, 4) dtype=int16_ref>,
        #  'page_map': <tf.Variable 'page_map:0' shape=(52752, 4) dtype=int32_ref>,
        #  'page_ix': <tf.Variable 'page_ix:0' shape=(145036,) dtype=string_ref>,
        #  'pf_agent': <tf.Variable 'pf_agent:0' shape=(145036, 4) dtype=float32_ref>,
        #  'pf_country': <tf.Variable 'pf_country:0' shape=(145036, 7) dtype=float32_ref>,
        #  'pf_site': <tf.Variable 'pf_site:0' shape=(145036, 3) dtype=float32_ref>,
        #  'page_popularity': <tf.Variable 'page_popularity:0' shape=(145036,) dtype=float32_ref>,
        #  'year_autocorr': <tf.Variable 'year_autocorr:0' shape=(145036,) dtype=float32_ref>,
        #  'quarter_autocorr': <tf.Variable 'quarter_autocorr:0' shape=(145036,) dtype=float32_ref>,
        #  'dow': <tf.Variable 'dow:0' shape=(867, 2) dtype=float32_ref>,
        #  'features_days': 867, 'data_days': 805, 'n_pages': 145036, 'data_start': '2015-07-01',
        #  'data_end': Timestamp('2017-09-11 00:00:00'), 'features_end': Timestamp('2017-11-13 00:00:00')}
        # print(f"variables 1 ={variables}; plain_vars={plain_vars}")
        if plain_vars:
            # todo 这里update是增加key,value
            variables.update(plain_vars)
        # print(f"variables 2 ={variables}")
        super().__init__(variables)
        self.path = path
        self.saver = tf.train.Saver(tensors, name='varfeeder_saver')
        for var in variables:
            if var not in self.__dict__:
                self.__dict__[var] = variables[var]

    def restore(self, session):
        """
        Restores variable content
        :param session: current session
        :return: variable list
        """
        self.saver.restore(session, os.path.join(self.path, 'feeder.cpt'))
        return self

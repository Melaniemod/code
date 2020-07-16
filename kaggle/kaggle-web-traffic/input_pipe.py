import tensorflow as tf

from feeder import VarFeeder
from enum import Enum
from typing import List, Iterable
import numpy as np
import pandas as pd
from datetime import datetime

class ModelMode(Enum):
    TRAIN = 0
    EVAL = 1,
    PREDICT = 2


class Split:
    def __init__(self, test_set: List[tf.Tensor], train_set: List[tf.Tensor], test_size: int, train_size: int):
        self.test_set = test_set
        self.train_set = train_set
        self.test_size = test_size
        self.train_size = train_size


class Splitter:
    def cluster_pages(self, cluster_idx: tf.Tensor):
        """
        Shuffles pages so all user_agents of each unique pages stays together in a shuffled list
        :param cluster_idx: Tensor[uniq_pages, n_agents], each value is index of pair (uniq_page, agent) in other page tensors
        :return: list of page indexes for use in a global page tensors
        """
        size = cluster_idx.shape[0].value
        # 这样做shutter可以保障同一个事件的样本是在一起的。
        random_idx = tf.random_shuffle(tf.range(0, size, dtype=tf.int32), self.seed)
        shuffled_pages = tf.gather(cluster_idx, random_idx)
        # Drop non-existent (uniq_page, agent) pairs. Non-existent pair has index value = -1
        mask = shuffled_pages >= 0
        # todo 最后输出一维的list
        page_idx = tf.boolean_mask(shuffled_pages, mask)
        return page_idx

    def __init__(self, tensors: List[tf.Tensor], cluster_indexes: tf.Tensor, n_splits, seed, train_sampling=1.0,
                 test_sampling=1.0):
        size = tensors[0].shape[0].value
        self.seed = seed
        clustered_index = self.cluster_pages(cluster_indexes)
        index_len = tf.shape(clustered_index)[0]
        assert_op = tf.assert_equal(index_len, size, message='n_pages is not equals to size of clustered index')
        with tf.control_dependencies([assert_op]):
            split_nitems = int(round(size / n_splits))
            split_size = [split_nitems] * n_splits
            split_size[-1] = size - (n_splits - 1) * split_nitems
            splits = tf.split(clustered_index, split_size)
            # 因为有相同的seed，所以，shutter的结果相同
            complements = [tf.random_shuffle(tf.concat(splits[:i] + splits[i + 1:], axis=0), seed) for i in
                           range(n_splits)]
            splits = [tf.random_shuffle(split, seed) for split in splits]
            # todo complements,splits
            #  [<tf.Tensor 'RandomShuffle_1:0' shape=(96691,) dtype=int32>,
            #  <tf.Tensor 'RandomShuffle_2:0' shape=(96691,) dtype=int32>,
            #  <tf.Tensor 'RandomShuffle_3:0' shape=(96690,) dtype=int32>] ===
            #  [<tf.Tensor 'RandomShuffle_4:0' shape=(48345,) dtype=int32>,
            #  <tf.Tensor 'RandomShuffle_5:0' shape=(48345,) dtype=int32>,
            #  <tf.Tensor 'RandomShuffle_6:0' shape=(48346,) dtype=int32>]
            # print("complements,splits",complements,'===\n',splits)


        def mk_name(prefix, tensor):
            return prefix + '_' + tensor.name[:-2]

        def prepare_split(i):
            test_size = split_size[i]
            train_size = size - test_size
            test_sampled_size = int(round(test_size * test_sampling))
            train_sampled_size = int(round(train_size * train_sampling))
            test_idx = splits[i][:test_sampled_size]
            train_idx = complements[i][:train_sampled_size]
            test_set = [tf.gather(tensor, test_idx, name=mk_name('test', tensor)) for tensor in tensors]
            tran_set = [tf.gather(tensor, train_idx, name=mk_name('train', tensor)) for tensor in tensors]
            return Split(test_set, tran_set, test_sampled_size, train_sampled_size)

        self.splits = [prepare_split(i) for i in range(n_splits)]


class FakeSplitter:
    def __init__(self, tensors: List[tf.Tensor], n_splits, seed, test_sampling=1.0):
        # todo 输入为：(inp.hits, inp.pf_agent, inp.pf_country, inp.pf_site,
        #             inp.page_ix, inp.page_popularity, inp.year_autocorr, inp.quarter_autocorr)
        total_pages = tensors[0].shape[0].value
        n_pages = int(round(total_pages * test_sampling))

        def mk_name(prefix, tensor):
            return prefix + '_' + tensor.name[:-2]

        def prepare_split(i):
            idx = tf.random_shuffle(tf.range(0, n_pages, dtype=tf.int32), seed + i)
            train_tensors = [tf.gather(tensor, idx, name=mk_name('shfl', tensor)) for tensor in tensors]
            if test_sampling < 1.0:
                sampled_idx = idx[:n_pages]
                test_tensors = [tf.gather(tensor, sampled_idx, name=mk_name('shfl_test', tensor)) for tensor in tensors]
            else:
                test_tensors = train_tensors
                # todo total_pages=145036; n_pages=145036; test_sampling=1.0;
                #  train_tensors=[<tf.Tensor 'shfl_hits_2:0' shape=(145036, 805) dtype=float32>,
                #  <tf.Tensor 'shfl_pf_agent_2:0' shape=(145036, 4) dtype=float32>,
                #  <tf.Tensor 'shfl_pf_country_2:0' shape=(145036, 7) dtype=float32>,
                #  <tf.Tensor 'shfl_pf_site_2:0' shape=(145036, 3) dtype=float32>,
                #  <tf.Tensor 'shfl_page_ix_2:0' shape=(145036,) dtype=string>,
                #  <tf.Tensor 'shfl_page_popularity_2:0' shape=(145036,) dtype=float32>,
                #  <tf.Tensor 'shfl_year_autocorr_2:0' shape=(145036,) dtype=float32>,
                #  <tf.Tensor 'shfl_quarter_autocorr_2:0' shape=(145036,) dtype=float32>];
                #  test_tensors=[<tf.Tensor 'shfl_hits_2:0' shape=(145036, 805) dtype=float32>,
                #  <tf.Tensor 'shfl_pf_agent_2:0' shape=(145036, 4) dtype=float32>,
                #  <tf.Tensor 'shfl_pf_country_2:0' shape=(145036, 7) dtype=float32>,
                #  <tf.Tensor 'shfl_pf_site_2:0' shape=(145036, 3) dtype=float32>,
                #  <tf.Tensor 'shfl_page_ix_2:0' shape=(145036,) dtype=string>,
                #  <tf.Tensor 'shfl_page_popularity_2:0' shape=(145036,) dtype=float32>,
                #  <tf.Tensor 'shfl_year_autocorr_2:0' shape=(145036,) dtype=float32>,
                #  <tf.Tensor 'shfl_quarter_autocorr_2:0' shape=(145036,) dtype=float32>]
                # print(f"total_pages={total_pages}; n_pages={n_pages}; test_sampling={test_sampling}; "
                #       f" train_tensors={train_tensors}; test_tensors={test_tensors}")
            # todo 这里的训练集和测试集是一样的？返回的是 测试集、训练集，测试集网页数量，全量网页数量。
            #  side_split=False；forward_split=False的时候这个test_tensor 应该是没用用到的
            #  test_tensors=[<tf.Tensor 'shfl_hits_2:0' shape=(145036, 805) dtype=float32>,
            #  <tf.Tensor 'shfl_pf_agent_2:0' shape=(145036, 4) dtype=float32>,
            #  <tf.Tensor 'shfl_pf_country_2:0' shape=(145036, 7) dtype=float32>,
            #  <tf.Tensor 'shfl_pf_site_2:0' shape=(145036, 3) dtype=float32>,
            #  <tf.Tensor 'shfl_page_ix_2:0' shape=(145036,) dtype=string>,
            #  <tf.Tensor 'shfl_page_popularity_2:0' shape=(145036,) dtype=float32>,
            #  <tf.Tensor 'shfl_year_autocorr_2:0' shape=(145036,) dtype=float32>,
            #  <tf.Tensor 'shfl_quarter_autocorr_2:0' shape=(145036,) dtype=float32>],
            #  train_tensors=[<tf.Tensor 'shfl_hits_2:0' shape=(145036, 805) dtype=float32>,
            #  <tf.Tensor 'shfl_pf_agent_2:0' shape=(145036, 4) dtype=float32>,
            #  <tf.Tensor 'shfl_pf_country_2:0' shape=(145036, 7) dtype=float32>,
            #  <tf.Tensor 'shfl_pf_site_2:0' shape=(145036, 3) dtype=float32>,
            #  <tf.Tensor 'shfl_page_ix_2:0' shape=(145036,) dtype=string>,
            #  <tf.Tensor 'shfl_page_popularity_2:0' shape=(145036,) dtype=float32>,
            #  <tf.Tensor 'shfl_year_autocorr_2:0' shape=(145036,) dtype=float32>,
            #  <tf.Tensor 'shfl_quarter_autocorr_2:0' shape=(145036,) dtype=float32>],
            #  n_pages=145036, total_pages=145036
            # print(f"test_tensors={test_tensors}, train_tensors={train_tensors}, n_pages={n_pages}, total_pages={total_pages}")
            return Split(test_tensors, train_tensors, n_pages, total_pages)

        self.splits = [prepare_split(i) for i in range(n_splits)]


class InputPipe:
    def cut(self, hits, start, end):
        """
        Cuts [start:end] diapason from input data
        todo 对 hit(网页访问量) 先截取，后分为x,y（应该是encode和decode对应的点击量吧？）
         dow 周几的 正弦余弦 截取
         lagged_ix 滑窗下标 根据滑窗的下标去相应的点击量
        :param hits: hits timeseries
        :param start: start index
        :param end: end index
        :return: tuple (train_hits, test_hits, dow, lagged_hits)
        """
        # Pad hits to ensure we have enough array length for prediction
        # todo 这里增加一个预测窗口的长度是为什么？
        #  hits=Tensor("args_0:0", shape=(805,), dtype=float32);
        #  start=Tensor("random_uniform:0", shape=(), dtype=int32);
        #  end=Tensor("add:0", shape=(), dtype=int32)
        # print(f"hits={hits}; start={start},end={end}")
        hits = tf.concat([hits, tf.fill([self.predict_window], np.NaN)], axis=0)
        # todo hits=Tensor("concat:0", shape=(868,), dtype=float32)
        # print(f"hits={hits}")
        cropped_hit = hits[start:end]

        # cut day of week   dow:星期的正弦余弦
        # todo 这里使用了 inp.dow 为什么 hits 不用
        cropped_dow = self.inp.dow[start:end]

        # Cut lagged hits
        # todo 下面到空行之前都是根据滑窗的 index 取滑窗的访问量
        # gather() 抽取第几维
        # tf.cast：用于改变某个张量的数据类型；    lagged_ix：对起止时间做3，6，9，12个月的滑窗
        cropped_lags = tf.cast(self.inp.lagged_ix[start:end], tf.int32)
        # Mask for -1 (no data) lag indexes
        lag_mask = cropped_lags < 0
        # todo lag_mask=Tensor("Less:0", shape=(?, 4), dtype=bool)。最早的数据滑窗的时候应该会有<0的值；这里<0的直接去第一个时间步的是吧？
        #  后面 lagged_hit 对<0的直接置为0了
        # print(f'lag_mask={lag_mask}')
        # Convert -1 to 0 for gather(), it don't accept anything exotic
        cropped_lags = tf.maximum(cropped_lags, 0)
        # Translate lag indexes to hit values
        lagged_hit = tf.gather(hits, cropped_lags)
        # todo cropped_lags=Tensor("Maximum:0", shape=(?, 4), dtype=int32)
        #  lag_mask=Tensor("Less:0", shape=(?, 4), dtype=bool)
        #  lagged_hit=Tensor("GatherV2:0", shape=(?, 4), dtype=float32)
        # print("cropped_lags,lag_mask,lagged_hit",cropped_lags,lag_mask,lagged_hit)
        # Convert masked (see above) or NaN lagged hits to zeros
        # todo tf.zeros_like函数返回将所有元素设置为零的张量.
        lag_zeros = tf.zeros_like(lagged_hit)
        lagged_hit = tf.where(lag_mask | tf.is_nan(lagged_hit), lag_zeros, lagged_hit)

        # Split for train and test
        # x_hite,y_hite分别是什么？
        # todo cropped_hit=Tensor("strided_slice:0", shape=(?,),dtype=float32)
        #  ,self.train_window=283 ,
        #  self.predict_window=63
        # print(f"cropped_hit={cropped_hit},self.train_window={self.train_window}, self.predict_window={self.predict_window}")
        # todo 这个函数的用途简单说就是把一个张量划分成几个子张量。其中分割方式分为两种
        #  1. 如果num_or_size_splits 传入的 是一个整数，那直接在axis=D这个维度上把张量平均切分成几个小张量
        #  2. 如果num_or_size_splits 传入的是一个向量（这里向量各个元素的和要跟原本这个维度的数值相等）就根据这个向量有几个元素分为几项）
        x_hits, y_hits = tf.split(cropped_hit, [self.train_window, self.predict_window], axis=0)

        # Convert NaN to zero in for train data
        x_hits = tf.where(tf.is_nan(x_hits), tf.zeros_like(x_hits), x_hits)
        # todo x_hits, y_hits, cropped_dow, lagged_hit
        #  Tensor("Select_1:0", shape=(283,), dtype=float32)
        #  Tensor("split:1", shape=(63,), dtype=float32)
        #  Tensor("strided_slice_1:0", shape=(?, 2), dtype=float32)
        #  Tensor("Select:0", shape=(?, 4), dtype=float32)
        # print("x_hits, y_hits, cropped_dow, lagged_hit",x_hits, y_hits, cropped_dow, lagged_hit)
        return x_hits, y_hits, cropped_dow, lagged_hit

    def cut_train(self, hits, *args):
        """
        Cuts a segment of time series for training. Randomly chooses starting point.
        :param hits: hits timeseries
        :param args: pass-through data, will be appended to result
        :return: result of cut() + args
        """
        # todo *args=
        #  (<tf.Tensor 'args_1:0' shape=(4,) dtype=float32>,
        #  <tf.Tensor 'args_2:0' shape=(7,) dtype=float32>,
        #  <tf.Tensor 'args_3:0' shape=(3,) dtype=float32>,
        #  <tf.Tensor 'args_4:0' shape=() dtype=string>,
        #  <tf.Tensor 'args_5:0' shape=() dtype=float32>,
        #  <tf.Tensor 'args_6:0' shape=() dtype=float32>,
        #  <tf.Tensor 'args_7:0' shape=() dtype=float32>)
        # print(f"*args={args}")
        n_days = self.predict_window + self.train_window
        # How much free space we have to choose starting day
        # todo self.inp.data_days=805, n_days=346, self.back_offset=0, self.start_offset=0
        # print(f"self.inp.data_days={self.inp.data_days}, n_days={n_days}, self.back_offset={self.back_offset}, self.start_offset={self.start_offset}")
        # todo 整体数据日期 - 一个训练序列要用到的encoding的长度和decoding的长度 - 开始不用的长度 - 结尾不用的长度
        #  这里cut是指定一个日期，根据这个日期截取 train_window + predict_window 长度的序列。这里的 free_space 就是这个日期的最晚的值
        free_space = self.inp.data_days - n_days - self.back_offset - self.start_offset
        if self.verbose:
            # tmp1 = pd.Timedelta(self.start_offset, 'D')
            # tmp2 = self.inp.data_start
            # print("start_offset === ",'\n',tmp1,type(tmp1),'\n',tmp2,type(self.inp.data_start))

            lower_train_start = datetime.strptime(self.inp.data_start,'%Y-%m-%d') + pd.Timedelta(self.start_offset, 'D')
            lower_test_end = lower_train_start + pd.Timedelta(n_days, 'D')
            lower_test_start = lower_test_end - pd.Timedelta(self.predict_window, 'D')
            upper_train_start = datetime.strptime(self.inp.data_start,'%Y-%m-%d') + pd.Timedelta(free_space - 1, 'D')
            upper_test_end = upper_train_start + pd.Timedelta(n_days, 'D')
            upper_test_start = upper_test_end - pd.Timedelta(self.predict_window, 'D')
            # todo Free space for training: 459 days.
            #  Lower train 2015-07-01 00:00:00, prediction 2016-04-09 00:00:00..2016-06-11 00:00:00
            #  Upper train 2016-10-01 00:00:00, prediction 2017-07-11 00:00:00..2017-09-12 00:00:00
            print(f"Free space for training: {free_space} days.")
            # print(f" Lower train {lower_train_start}, prediction {lower_test_start}..{lower_test_end}")
            # print(f" Upper train {upper_train_start}, prediction {upper_test_start}..{upper_test_end}")
        # Random starting point
        # todo tf.random_uniform((4, 4), minval=low,maxval=high,dtype=tf.float32)))返回4*4的矩阵，产生于low和high之间，产生的值是均匀分布的。
        offset = tf.random_uniform((), self.start_offset, free_space, dtype=tf.int32, seed=self.rand_seed)
        end = offset + n_days
        # Cut all the things
        # todo 只cut了3次
        # print(f"-------cut-------")
        return self.cut(hits, offset, end) + args

    def cut_eval(self, hits, *args):
        """
        # todo 训练的时候起点是随机的，交叉验证集上的时候是固定的，这样不会存在
        Cuts segment of time series for evaluation.
        Always cuts train_window + predict_window length segment beginning at start_offset point
        :param hits: hits timeseries
        :param args: pass-through data, will be appended to result
        :return: result of cut() + args
        """
        end = self.start_offset + self.train_window + self.predict_window
        return self.cut(hits, self.start_offset, end) + args

    def reject_filter(self, x_hits, y_hits, *args):
        """
        Rejects timeseries having too many zero datapoints (more than self.max_train_empty)
        """
        if self.verbose:
            # todo max empty 280 train 62 predict
            print("max empty %d train %d predict" % (self.max_train_empty, self.max_predict_empty))
        zeros_x = tf.reduce_sum(tf.to_int32(tf.equal(x_hits, 0.0)))
        keep = zeros_x <= self.max_train_empty
        return keep

    def make_features(self, x_hits, y_hits, dow, lagged_hits, pf_agent, pf_country, pf_site, page_ix,
                      page_popularity, year_autocorr, quarter_autocorr):
        """
        todo 根据数据处理成特征。数据分为几类：时间、网页ID：x_hits，x_lagged; 时间序列：x_dow；网页特征：page_features
        Main method. Assembles input data into final tensors
        """
        # Split day of week to train and test
        # todo dow sin cos之后的周几
        x_dow, y_dow = tf.split(dow, [self.train_window, self.predict_window], axis=0)
        # todo x_dow=Tensor("split:0", shape=(283, 2), dtype=float32);
        #  y_dow=Tensor("split:1", shape=(63, 2), dtype=float32);
        #  dow=Tensor("args_2:0", shape=(?, 2), dtype=float32)
        # print(f"x_dow={x_dow}; y_dow={y_dow}; dow={dow}")

        # Normalize hits
        # todo 访问量归一化
        mean = tf.reduce_mean(x_hits)
        std = tf.sqrt(tf.reduce_mean(tf.squared_difference(x_hits, mean)))
        norm_x_hits = (x_hits - mean) / std
        norm_y_hits = (y_hits - mean) / std
        norm_lagged_hits = (lagged_hits - mean) / std

        # Split lagged hits to train and test
        x_lagged, y_lagged = tf.split(norm_lagged_hits, [self.train_window, self.predict_window], axis=0)

        # Combine all page features into single tensor
        # todo 这些都是不随着时间变化的特征，访问量是随着时间序列变化的特征
        stacked_features = tf.stack([page_popularity, quarter_autocorr, year_autocorr])
        flat_page_features = tf.concat([pf_agent, pf_country, pf_site, stacked_features], axis=0)
        page_features = tf.expand_dims(flat_page_features, 0)

        # Train features
        x_features = tf.concat([
            # todo [n_days] -> [n_days, 1]；比输入张量多1维但是包含相同数据的张量。tf.expand_dims() 方法的反向操作为 tf.squeeze() 方法
            tf.expand_dims(norm_x_hits, -1),
            x_dow,
            x_lagged,
            # Stretch page_features to all training days
            # [1, features] -> [n_days, features]
            # todo 平铺之意，用于在同一维度上的复制
            tf.tile(page_features, [self.train_window, 1])
        ], axis=1)
        # todo tf.expand_dims(norm_x_hits, -1) : Tensor("ExpandDims_2:0", shape=(283, 1), dtype=float32);
        #  norm_x_hits: Tensor("truediv:0", shape=(283,), dtype=float32);
        #  x_dow : Tensor("split:0", shape=(283, 2), dtype=float32);
        #  x_lagged : Tensor("split_1:0", shape=(283, 4), dtype=float32);
        #  tf.tile(page_features, [self.train_window, 1]): Tensor("Tile_1:0", shape=(283, 17), dtype=float32);
        #  page_features:Tensor("ExpandDims:0", shape=(1, 17), dtype=float32)
        # print(f"tf.expand_dims(norm_x_hits, -1) : {tf.expand_dims(norm_x_hits, -1)}; norm_x_hits: {norm_x_hits}; "
        #       f"x_dow : {x_dow}; x_lagged : {x_lagged}; tf.tile(page_features, [self.train_window, 1]): "
        #       f"{tf.tile(page_features, [self.train_window, 1])}; page_features:{page_features}")

        # Test features
        y_features = tf.concat([
            # [n_days] -> [n_days, 1]
            y_dow,
            y_lagged,
            # Stretch page_features to all testing days
            # [1, features] -> [n_days, features]
            tf.tile(page_features, [self.predict_window, 1])
        ], axis=1)

        return x_hits, x_features, norm_x_hits, x_lagged, y_hits, y_features, norm_y_hits, mean, std, flat_page_features, page_ix

    def __init__(self, inp: VarFeeder, features: Iterable[tf.Tensor], n_pages: int, mode: ModelMode, n_epoch=None,
                 batch_size=127, runs_in_burst=1, verbose=True, predict_window=60, train_window=500,
                 train_completeness_threshold=1, predict_completeness_threshold=1, back_offset=0,
                 train_skip_first=0, rand_seed=None):
        """
        Create data preprocessing pipeline
        :param inp: Raw input data
        :param features: Features tensors (subset of data in inp)
        :param n_pages: Total number of pages
        :param mode: Train/Predict/Eval mode selector
        :param n_epoch: Number of epochs. Generates endless data stream if None
        :param batch_size:
        :param runs_in_burst: How many batches can be consumed at short time interval (burst). Multiplicator for prefetch()
        :param verbose: Print additional information during graph construction
        :param predict_window: Number of days to predict
        :param train_window: Use train_window days for traning
        :param train_completeness_threshold: Percent of zero datapoints allowed in train timeseries.
        :param predict_completeness_threshold: Percent of zero datapoints allowed in test/predict timeseries.
        :param back_offset: Don't use back_offset days at the end of timeseries
        :param train_skip_first: Don't use train_skip_first days at the beginning of timeseries
        todo 为啥要有 train_skip_first 和 back_offset？？是开始几天和结尾几天不用对吧？
         batch_size=128,
        :param rand_seed:
        """
        self.n_pages = n_pages
        self.inp = inp
        self.batch_size = batch_size
        self.rand_seed = rand_seed
        self.back_offset = back_offset
        if verbose:
            # todo Mode:ModelMode.TRAIN, data days:805, Data start:2015-07-01,
            #  data end:2017-09-11 00:00:00, features end:2017-11-13 00:00:00
            print("Mode:%s, data days:%d, Data start:%s, data end:%s, features end:%s " % (
            mode, inp.data_days, inp.data_start, inp.data_end, inp.features_end))

        if mode == ModelMode.TRAIN:
            # reserve predict_window at the end for validation
            assert inp.data_days - predict_window > predict_window + train_window, \
                "Predict+train window length (+predict window for validation) is larger than total number of days in dataset"
            self.start_offset = train_skip_first
        elif mode == ModelMode.EVAL or mode == ModelMode.PREDICT:
            self.start_offset = inp.data_days - train_window - back_offset
            if verbose:
                train_start = inp.data_start + pd.Timedelta(self.start_offset, 'D')
                eval_start = train_start + pd.Timedelta(train_window, 'D')
                end = eval_start + pd.Timedelta(predict_window - 1, 'D')
                print("Train start %s, predict start %s, end %s" % (train_start, eval_start, end))
            assert self.start_offset >= 0

        self.train_window = train_window
        self.predict_window = predict_window
        # todo 这里是干什么呢？ train_window = 283；predict_window = 63
        self.attn_window = train_window - predict_window + 1
        # todo 这里 1 - train_completeness_threshold 不是非0的最小数量吧。
        self.max_train_empty = int(round(train_window * (1 - train_completeness_threshold)))
        self.max_predict_empty = int(round(predict_window * (1 - predict_completeness_threshold)))
        # todo train_completeness_threshold=0.01; predict_completeness_threshold=0.01
        # print(f"train_completeness_threshold={train_completeness_threshold}; predict_completeness_threshold={predict_completeness_threshold}")
        self.mode = mode
        self.verbose = verbose

        # Reserve more processing threads for eval/predict because of larger batches
        num_threads = 3 if mode == ModelMode.TRAIN else 6

        # Choose right cutter function for current ModelMode
        cutter = {ModelMode.TRAIN: self.cut_train, ModelMode.EVAL: self.cut_eval, ModelMode.PREDICT: self.cut_eval}
        # todo https://blog.csdn.net/u014061630/article/details/80728694
        #  tf.data.Dataset：表示一系列元素，其中每个元素包含一个或多个 Tensor 对象。
        #  例如，在图片管道中，一个元素可能是单个训练样本，具有一对表示图片数据和标签的张量。可以通过两种不同的方式来创建数据集。
        #  1、直接从 Tensor 创建 Dataset（例如 Dataset.from_tensor_slices()）；当然 Numpy 也是可以的，TensorFlow 会自动将其转换为 Tensor。
        #  2、通过对一个或多个 tf.data.Dataset 对象来使用变换（例如 Dataset.batch()）来创建 Dataset。
        #  tf.data.Iterator：这是从数据集中提取元素的主要方法。Iterator.get_next() 指令会在执行时生成 Dataset 的下一个元素，
        #  并且此指令通常充当输入管道和模型之间的接口。最简单的迭代器是“单次迭代器”，它会对处理好的 Dataset 进行单次迭代。
        #  要实现更复杂的用途，您可以通过 `Iterator.initializer` 指令使用不同的数据集重新初始化和参数化迭代器，这样一来，就可以在同一个程序中对训练和验证数据进行多次迭代。
        #  要构建输入 pipeline，你必须首先根据数据集的存储方式选择相应的方法创建 Dataset 对象来读取数据。
        #  有了 Dataset 对象以后，您就可以通过使用 tf.data.Dataset 对象的各种方法对其进行处理。
        #  例如，您可以对Dataset的每一个元素使用某种变换，例 Dataset.map()（为每个元素使用一个函数），也可以对多个元素使用某种变换（例如 Dataset.batch()）
        root_ds = tf.data.Dataset.from_tensor_slices(tuple(features)).repeat(n_epoch)
        # Create dataset, transform features and assemble batches
        # todo features=[<tf.Tensor 'shfl_hits_2:0' shape=(145036, 805) dtype=float32>,
        #  <tf.Tensor 'shfl_pf_agent_2:0' shape=(145036, 4) dtype=float32>,
        #  <tf.Tensor 'shfl_pf_country_2:0' shape=(145036, 7) dtype=float32>,
        #  <tf.Tensor 'shfl_pf_site_2:0' shape=(145036, 3) dtype=float32>,
        #  <tf.Tensor 'shfl_page_ix_2:0' shape=(145036,) dtype=string>,
        #  <tf.Tensor 'shfl_page_popularity_2:0' shape=(145036,) dtype=float32>,
        #  <tf.Tensor 'shfl_year_autocorr_2:0' shape=(145036,) dtype=float32>,
        #  <tf.Tensor 'shfl_quarter_autocorr_2:0' shape=(145036,) dtype=float32>];
        #  root_ds=<DatasetV1Adapter shapes: ((805,), (4,), (7,), (3,), (), (), (), ()), types: (tf.float32, tf.float32, tf.float32, tf.float32, tf.string, tf.float32, tf.float32, tf.float32)>;
        #  n_epoch=None；
        #  tuple(features)=(<tf.Tensor 'shfl_hits_2:0' shape=(145036, 805) dtype=float32>,
        #  <tf.Tensor 'shfl_pf_agent_2:0' shape=(145036, 4) dtype=float32>,
        #  <tf.Tensor 'shfl_pf_country_2:0' shape=(145036, 7) dtype=float32>,
        #  <tf.Tensor 'shfl_pf_site_2:0' shape=(145036, 3) dtype=float32>,
        #  <tf.Tensor 'shfl_page_ix_2:0' shape=(145036,) dtype=string>,
        #  <tf.Tensor 'shfl_page_popularity_2:0' shape=(145036,) dtype=float32>,
        #  <tf.Tensor 'shfl_year_autocorr_2:0' shape=(145036,) dtype=float32>,
        #  <tf.Tensor 'shfl_quarter_autocorr_2:0' shape=(145036,) dtype=float32>)
        # print(F"features={features}; root_ds={root_ds}; n_epoch={n_epoch}； tuple(features)={tuple(features)}")
        #  todo 方法prefetch(buffer_size) 参数 buffer_size 代表将被加入缓冲器的元素的最大数。
        #   https: // blog.csdn.net / Eartha1995 / article / details / 84930492
        #   buffer_size会影响dataset的随机性，即元素生成的顺序。Dataset.prefetch（）中的buffer_size仅仅影响生成下一个元素的时间。
        #   加一个prefetch buffer能够提高性能，通过将数据预处理与下游计算重叠。典型地，在管道末尾增加一个prefetch buffer（也许仅仅是单个样本），但更复杂的管道能够从额外的prefetching获益，尤其是当生成单个元素的时间变化时。
        #   上面 -----cut-----虽然只调用了3次，但是第一次map之后会偷偷调用cut 吗？应该不会偷偷调用. 有可能会像生成器一样虽然创建的时候只创建一次，但是创建完之后可以可以从中多次取元素

        # todo 如果.repeat(n_epoch)改为.repeat(1)的化则为下面这样子
        #  features=[<tf.Tensor 'shfl_hits:0' shape=(145036, 805) dtype=float32>,
        #  <tf.Tensor 'shfl_pf_agent:0' shape=(145036, 4) dtype=float32>,
        #  <tf.Tensor 'shfl_pf_country:0' shape=(145036, 7) dtype=float32>,
        #  <tf.Tensor 'shfl_pf_site:0' shape=(145036, 3) dtype=float32>,
        #  <tf.Tensor 'shfl_page_ix:0' shape=(145036,) dtype=string>,
        #  <tf.Tensor 'shfl_page_popularity:0' shape=(145036,) dtype=float32>,
        #  <tf.Tensor 'shfl_year_autocorr:0' shape=(145036,) dtype=float32>,
        #  <tf.Tensor 'shfl_quarter_autocorr:0' shape=(145036,) dtype=float32>];
        #  root_ds=<DatasetV1Adapter shapes: ((805,), (4,), (7,), (3,), (), (), (), ()), types: (tf.float32, tf.float32, tf.float32, tf.float32, tf.string, tf.float32, tf.float32, tf.float32)>;
        #  n_epoch=None；
        #  tuple(features)=(<tf.Tensor 'shfl_hits:0' shape=(145036, 805) dtype=float32>,
        #  <tf.Tensor 'shfl_pf_agent:0' shape=(145036, 4) dtype=float32>,
        #  <tf.Tensor 'shfl_pf_country:0' shape=(145036, 7) dtype=float32>,
        #  <tf.Tensor 'shfl_pf_site:0' shape=(145036, 3) dtype=float32>,
        #  <tf.Tensor 'shfl_page_ix:0' shape=(145036,) dtype=string>,
        #  <tf.Tensor 'shfl_page_popularity:0' shape=(145036,) dtype=float32>,
        #  <tf.Tensor 'shfl_year_autocorr:0' shape=(145036,) dtype=float32>,
        #  <tf.Tensor 'shfl_quarter_autocorr:0' shape=(145036,) dtype=float32>)
        # todo 这里 map(cut) 应该是指对每个网站(或者说每个样本)进行 cut，这样cut之后不同的网站的起止时间应该是不一样的
        batch = (root_ds
                 .map(cutter[mode])
                 .filter(self.reject_filter)
                 .map(self.make_features, num_parallel_calls=num_threads)
                 .batch(batch_size)
                 .prefetch(runs_in_burst * 2)
                 )
        # x_hits, x_features, norm_x_hits, x_lagged, y_hits, y_features, norm_y_hits, mean, std, flat_page_features, page_ix

        self.iterator = batch.make_initializable_iterator()
        # todo terator.get_next() 方法返回一个或多个 tf.Tensor 对象，这些对象对应于迭代器的下一个元素。
        #  每次 eval 这些张量时，它们都会获取底层数据集中下一个元素的值。（要注意 Iterator.get_next() 并不会立即使迭代器进入下个状态。
        #  必须使用 TensorFlow 表达式中返回的 tf.Tensor 对象，并将该表达式的结果传递到 tf.Session.run()，以获取下一个元素并使迭代器进入下个状态。）
        it_tensors = self.iterator.get_next()

        # Assign all tensors to class variables
        self.true_x, self.time_x, self.norm_x, self.lagged_x, self.true_y, self.time_y, self.norm_y, self.norm_mean, \
        self.norm_std, self.page_features, self.page_ix = it_tensors

        self.encoder_features_depth = self.time_x.shape[2].value

    def load_vars(self, session):
        self.inp.restore(session)

    def init_iterator(self, session):
        session.run(self.iterator.initializer)

# todo 首先通过 page_features 返回tensor的 hits, pf_agent, pf_country, pf_site,page_ix, page_popularity, year_autocorr, quarter_autocorr
#  然后通过 FakeSplitter 创建了创建了3份数据，每份包括：训练集，测试集，训练集大小，测试集大小。
#  然后通过 InputPipe，随机选择起始时间，截取特征片段。然后对切过的片段数据做特征



def page_features(inp: VarFeeder):
    # todo type(inp):<class 'feeder.FeederVars'>
    # print(f"type(inp):{type(inp)}")
    return (inp.hits, inp.pf_agent, inp.pf_country, inp.pf_site,
            inp.page_ix, inp.page_popularity, inp.year_autocorr, inp.quarter_autocorr)

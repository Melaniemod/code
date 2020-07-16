import os.path
import shutil
import sys
import numpy as np
import tensorflow as tf
from tqdm import trange
from typing import List, Tuple
import heapq
import logging
import pandas as pd
from enum import Enum
import os

from hparams import build_from_set, build_hparams
from feeder import VarFeeder
from input_pipe import InputPipe, ModelMode, Splitter,FakeSplitter, page_features
from model import Model
import argparse


log = logging.getLogger('trainer')

class Ema:
    def __init__(self, k=0.99):
        self.k = k
        self.state = None
        self.steps = 0

    def __call__(self, *args, **kwargs):
        v = args[0]
        self.steps += 1
        if self.state is None:
            self.state = v
        else:
            eff_k = min(1 - 1 / self.steps, self.k)
            self.state = eff_k * self.state + (1 - eff_k) * v
        return self.state


class Metric:
    def __init__(self, name: str, op, smoothness: float = None):
        self.name = name
        self.op = op
        self.smoother = Ema(smoothness) if smoothness else None
        self.epoch_values = []
        self.best_value = np.Inf
        self.best_step = 0
        self.last_epoch = -1
        self.improved = False
        self._top = []

    @property
    def avg_epoch(self):
        return np.mean(self.epoch_values)

    @property
    def best_epoch(self):
        return np.min(self.epoch_values)

    @property
    def last(self):
        return self.epoch_values[-1] if self.epoch_values else np.nan

    @property
    def top(self):
        return -np.mean(self._top)


    def update(self, value, epoch, step):
        if self.smoother:
            value = self.smoother(value)
        if epoch > self.last_epoch:
            self.epoch_values = []
            self.last_epoch = epoch
        self.epoch_values.append(value)
        if value < self.best_value:
            self.best_value = value
            self.best_step = step
            self.improved = True
        else:
            self.improved = False
        if len(self._top) >= 5:
            # todo heapq.heappushpop(heap,item) 将 item 放入堆中，然后弹出并返回 heap 的最小元素
            heapq.heappushpop(self._top, -value)
        else:
            # todo heapq.heappush(heap,item) 将 item 的值加入 heap 中，保持堆的不变性。
            heapq.heappush(self._top, -value)


class AggMetric:
    def __init__(self, metrics: List[Metric]):
        self.metrics = metrics

    def _mean(self, fun) -> float:
        # noinspection PyTypeChecker
        return np.mean([fun(metric) for metric in self.metrics])

    @property
    def avg_epoch(self):
        return self._mean(lambda m: m.avg_epoch)

    @property
    def best_epoch(self):
        return self._mean(lambda m: m.best_epoch)

    @property
    def last(self):
        return self._mean(lambda m: m.last)

    @property
    def top(self):
        return self._mean(lambda m: m.top)

    @property
    def improved(self):
        return np.any([metric.improved for metric in self.metrics])


class DummyMetric:
    @property
    def avg_epoch(self):
        return np.nan

    @property
    def best_epoch(self):
        return np.nan

    @property
    def last(self):
        return np.nan

    @property
    def top(self):
        return np.nan

    @property
    def improved(self):
        return False

    @property
    def metrics(self):
        return []


class Stage(Enum):
    TRAIN = 0
    EVAL_SIDE = 1
    EVAL_FRWD = 2
    EVAL_SIDE_EMA = 3
    EVAL_FRWD_EMA = 4


class ModelTrainerV2:
    def __init__(self, train_model: Model, eval: List[Tuple[Stage, Model]], model_no=0,
                 patience=None, stop_metric=None, summary_writer=None):
        self.train_model = train_model
        if eval:
            self.eval_stages, self.eval_models = zip(*eval)
        else:
            self.eval_stages, self.eval_models = [], []
        self.stopped = False
        self.model_no = model_no
        self.patience = patience
        self.best_metric = np.inf
        self.bad_epochs = 0
        # todo 似乎并没有代码对stop_metric做更改，那么应该一值是None
        self.stop_metric = stop_metric
        self.summary_writer = summary_writer

        def std_metrics(model: Model, smoothness):
            return [Metric('SMAPE', model.smape, smoothness), Metric('MAE', model.mae, smoothness)]

        self._metrics = {Stage.TRAIN: std_metrics(train_model, 0.9) + [Metric('GrNorm', train_model.glob_norm)]}
        # todo self._metrics：{<Stage.TRAIN: 0>: [<__main__.Metric object at 0x7fa9784bc4e0>, <__main__.Metric object at 0x7fa960759a58>, <__main__.Metric object at 0x7fa9607213c8>]}
        # print(f"self._metrics：{self._metrics}")
        # todo eval=[]
        # print(f"eval={eval}")
        for stage, model in eval:
            self._metrics[stage] = std_metrics(model, None)
        self.dict_metrics = {key: {metric.name: metric for metric in metrics} for key, metrics in self._metrics.items()}
        # todo dict_metrics={<Stage.TRAIN: 0>:
        #  {'SMAPE': <__main__.Metric object at 0x7f4ef85ad518>,
        #  'MAE': <__main__.Metric object at 0x7f4ef8072c88>,
        #  'GrNorm': <__main__.Metric object at 0x7f4ef0785a58>}
        #  }
        # print(f"dict_metrics={self.dict_metrics}")

    def init(self, sess):
        for model in list(self.eval_models) + [self.train_model]:
            print(f"V2-init=")
            model.inp.init_iterator(sess)

    @property
    def metrics(self):
        return self._metrics

    @property
    def train_ops(self):
        model = self.train_model
        # todo model.train_op=
        #  name: "m_1/group_deps"
        #  op: "NoOp"
        #  input: "^m_1/Adam"
        #  input: "^m_1/ExponentialMovingAverage"
        #  device: "/device:GPU:0"
        # print(f"model.train_op={model.train_op}")
        return [model.train_op]  # , model.summaries

    def metric_ops(self, key):
        return [metric.op for metric in self._metrics[key]]

    def process_metrics(self, key, run_results, epoch, step):
        metrics = self._metrics[key]
        summaries = []
        for result, metric in zip(run_results, metrics):
            metric.update(result, epoch, step)
            summaries.append(tf.Summary.Value(tag=f"{key.name}/{metric.name}_0", simple_value=result))
        return summaries

    def end_epoch(self):
        # todo epoch=0 self.stop_metric=None; self.best_metric=inf; self.bad_epochs=0; self.stopped=False
        # print(f"self.stop_metric={self.stop_metric}; self.best_metric={self.best_metric};"
        #       f" self.bad_epochs={self.bad_epochs}; self.stopped={self.stopped}")
        if self.stop_metric:
            best_metric = self.stop_metric(self.dict_metrics)# self.dict_metrics[Stage.EVAL_FRWD]['SMAPE'].avg_epoch
            if self.best_metric > best_metric:
                self.best_metric = best_metric
                self.bad_epochs = 0
            else:
                self.bad_epochs += 1
                if self.bad_epochs > self.patience:
                    self.stopped = True
        # todo epoch=0 2 -- self.stop_metric=None; self.best_metric=inf; self.bad_epochs=0; self.stopped=False
        # print(f"2 -- self.stop_metric={self.stop_metric}; self.best_metric={self.best_metric};"
        #       f" self.bad_epochs={self.bad_epochs}; self.stopped={self.stopped}")

class MultiModelTrainer:
    def __init__(self, trainers: List[ModelTrainerV2], inc_step_op,
                 misc_global_ops=None):
        self.trainers = trainers
        self.inc_step = inc_step_op
        self.global_ops = misc_global_ops or []
        self.eval_stages = trainers[0].eval_stages
        # todo self.eval_stages: []
        # print(f"self.eval_stages: {self.eval_stages}")
        # todo ls_stop=[False, False, False]
        # ls_stop = [trainers.stopped for trainers in self.trainers]
        # print(f"ls_stop={ls_stop}")

    def active(self):
        # ls_stop = [trainer.stopped for trainer in self.trainers]
        # print(f"ls_stop={ls_stop}")
        return [trainer for trainer in self.trainers if not trainer.stopped]

    def _metric_step(self, stage, initial_ops, sess: tf.Session, epoch: int, step=None, repeats=1, summary_every=1):
        ops = initial_ops
        offsets, lengths = [], []
        trainers = self.active()
        # todo 三个模型，原来的op包括：第几个step。然后加上 mae, smape, GrNorm, 3种损失函数的计算
        # todo ops=[<tf.Tensor 'AssignAdd:0' shape=() dtype=int64_ref>,
        #  <tf.Operation 'm_0/group_deps' type=NoOp>,
        #  <tf.Operation 'm_1/group_deps' type=NoOp>,
        #  <tf.Operation 'm_2/group_deps' type=NoOp>]
        # print(f"ops={ops}")
        for trainer in trainers:
            offsets.append(len(ops))
            metric_ops = trainer.metric_ops(stage)
            lengths.append(len(metric_ops))
            ops.extend(metric_ops)
        # todo 这里的 NoOp 是一个综合的操作；封装了很多参数的迭代操作。
        #  ops=[<tf.Tensor 'AssignAdd:0' shape=() dtype=int64_ref>,
        #  <tf.Operation 'm_0/group_deps' type=NoOp>,
        #  <tf.Operation 'm_1/group_deps' type=NoOp>,
        #  <tf.Operation 'm_2/group_deps' type=NoOp>,
        #  <tf.Tensor 'm_0/truediv_3:0' shape=() dtype=float32>,
        #  <tf.Tensor 'm_0/absolute_difference/value:0' shape=() dtype=float32>,
        #  <tf.Tensor 'm_0/global_norm/global_norm:0' shape=() dtype=float32>,
        #  1
        #  <tf.Tensor 'm_1/truediv_3:0' shape=() dtype=float32>,
        #  <tf.Tensor 'm_1/absolute_difference/value:0' shape=() dtype=float32>,
        #  <tf.Tensor 'm_1/global_norm/global_norm:0' shape=() dtype=float32>,
        #  <tf.Tensor 'm_2/truediv_3:0' shape=() dtype=float32>,
        #  <tf.Tensor 'm_2/absolute_difference/value:0' shape=() dtype=float32>,
        #  <tf.Tensor 'm_2/global_norm/global_norm:0' shape=() dtype=float32>],
        #  offsets=[4, 7, 10],
        #  lengths=[3, 3, 3]
        # print(f"ops={ops},offsets={offsets},lengths={lengths}")
        # todo repeats ==1;
        if repeats > 1:
            all_results = np.stack([np.array(sess.run(ops)) for _ in range(repeats)])
            results = np.mean(all_results, axis=0)
        else:
            results = sess.run(ops)
        if step is None:
            step = results[0]

        for trainer, offset, length in zip(trainers, offsets, lengths):
            chunk = results[offset: offset + length]
            # todo chunk=[0.58031756, 0.6197995, 1.1014143];
            #  results=[21, None, None, None, 0.43868905, 0.47811687, 0.2062241, 0.49504888, 0.55112034, 0.27088073, 0.4977056, 0.5314191, 0.34194475];
            #  offsets = [4, 7, 10];
            #  lengths=[3, 3, 3];
            #  epoch=0;
            #  step=1
            # print(f"chunk={chunk}; results={results}; offsets = {offsets}; lengths={lengths}; epoch={epoch}; step={step}")
            summaries = trainer.process_metrics(stage, chunk, epoch, step)
            if trainer.summary_writer and step > 200 and (step % summary_every == 0):
                summary = tf.Summary(value=summaries)
                trainer.summary_writer.add_summary(summary, global_step=step)
        return results

    def train_step(self, sess: tf.Session, epoch: int):
        # todo 第几个step
        ops = [self.inc_step] + self.global_ops
        # todo ops=[<tf.Tensor 'AssignAdd:0' shape=() dtype=int64_ref>]
        # print(f"ops={ops}")
        for trainer in self.active():
            # todo 参数指数滑动平均的迭代的操作
            ops.extend(trainer.train_ops)
        # todo ops==[<tf.Tensor 'AssignAdd:0' shape=() dtype=int64_ref>,
        #  <tf.Operation 'm_0/group_deps' type=NoOp>,
        #  <tf.Operation 'm_1/group_deps' type=NoOp>,
        #  <tf.Operation 'm_2/group_deps' type=NoOp>]
        # print(f"ops=={ops}")
        results = self._metric_step(Stage.TRAIN, ops, sess, epoch, summary_every=20)
        #return results[:len(self.global_ops) + 1] # step, grad_norm
        # todo result[0]表示第几step
        return results[0]

    def eval_step(self, sess: tf.Session, epoch: int, step, n_batches, stages:List[Stage]=None):
        target_stages = stages if stages is not None else self.eval_stages
        # print(f"target_stages= {target_stages}")
        for stage in target_stages:
            self._metric_step(stage, [], sess, epoch, step, repeats=n_batches)

    def metric(self, stage, name):
        return AggMetric([trainer.dict_metrics[stage][name] for trainer in self.trainers])

    def end_epoch(self):
        for trainer in self.active():
            # todo trainer.stop_metric=None
            # print(f"trainer.stop_metric={trainer.stop_metric}")
            trainer.end_epoch()

    def has_active(self):
        return len(self.active())


def train(name, hparams, multi_gpu=False, n_models=1, train_completeness_threshold=0.01,
          seed=None, logdir='data/logs', max_epoch=100, patience=2, train_sampling=1.0,
          eval_sampling=1.0, eval_memsize=5, gpu=0, gpu_allow_growth=False, save_best_model=False,
          forward_split=False, write_summaries=False, verbose=False, asgd_decay=None, tqdm=True,
          side_split=True, max_steps=None, save_from_step=None, do_eval=True, predict_window=63):

    eval_k = int(round(26214 * eval_memsize / n_models))
    eval_batch_size = int(
        eval_k / (hparams.rnn_depth * hparams.encoder_rnn_layers))  # 128 -> 1024, 256->512, 512->256
    eval_pct = 0.1
    batch_size = hparams.batch_size
    train_window = hparams.train_window
    # todo eval_k = 43690,eval_batch_size = 163,eval_pct = 0,batch_size = 128,train_window = 283
    # print("eval_k = %d,eval_batch_size = %d,eval_pct = %d,batch_size = %d,train_window = %d" %(eval_k,eval_batch_size,eval_pct,batch_size,train_window))
    tf.reset_default_graph()
    if seed:
        tf.set_random_seed(seed)

    with tf.device("/cpu:0"):
        inp = VarFeeder.read_vars("data/vars")
        # print("side_split = %d,train_sampling= %d,eval_sampling= %d,seed= %d" % (
        #     side_split,train_sampling,eval_sampling,seed),f"inp={inp}, side_split={side_split}; type(inp)={type(inp)}")
        # todo side_split = 0,train_sampling= 1,eval_sampling= 1,seed= 5
        #  inp={'hits': <tf.Variable 'hits:0' shape=(145036, 805) dtype=float32_ref>,
        #  'lagged_ix': <tf.Variable 'lagged_ix:0' shape=(867, 4) dtype=int16_ref>,
        #  'page_map': <tf.Variable 'page_map:0' shape=(52752, 4) dtype=int32_ref>,
        #  'page_ix': <tf.Variable 'page_ix:0' shape=(145036,) dtype=string_ref>,
        #  'pf_agent': <tf.Variable 'pf_agent:0' shape=(145036, 4) dtype=float32_ref>,
        #  'pf_country': <tf.Variable 'pf_country:0' shape=(145036, 7) dtype=float32_ref>,
        #  'pf_site': <tf.Variable 'pf_site:0' shape=(145036, 3) dtype=float32_ref>,
        #  'page_popularity': <tf.Variable 'page_popularity:0' shape=(145036,) dtype=float32_ref>,
        #  'year_autocorr': <tf.Variable 'year_autocorr:0' shape=(145036,) dtype=float32_ref>,
        #  'quarter_autocorr': <tf.Variable 'quarter_autocorr:0' shape=(145036,) dtype=float32_ref>,
        #  'dow': <tf.Variable 'dow:0' shape=(867, 2) dtype=float32_ref>,'features_days': 867,
        #  'data_days': 805, 'n_pages': 145036, 'data_start': '2015-07-01',
        #  'data_end': Timestamp('2017-09-11 00:00:00'), 'features_end': Timestamp('2017-11-13 00:00:00')}
        #  side_split=False;
        #  type(inp)=<class 'feeder.FeederVars'>;
        # if True:
        if side_split:
            splitter = Splitter(page_features(inp), inp.page_map, 3, train_sampling=train_sampling,
                                test_sampling=eval_sampling, seed=seed)
        else:
            splitter = FakeSplitter(page_features(inp), 3, seed=seed, test_sampling=eval_sampling)

    real_train_pages = splitter.splits[0].train_size
    real_eval_pages = splitter.splits[0].test_size

    items_per_eval = real_eval_pages * eval_pct
    eval_batches = int(np.ceil(items_per_eval / eval_batch_size))
    steps_per_epoch = real_train_pages // batch_size
    eval_every_step = int(round(steps_per_epoch * eval_pct))
    # todo real_train_pages = 145036,real_eval_pages= 145036,items_per_eval= 14503,eval_batches= 89,
    #  steps_per_epoch= 1133,eval_every_step= 113 -- 每个epoch有1133个step，每113个step打印一下当前模型的效果
    # print("real_train_pages = %d,real_eval_pages= %d,items_per_eval= %d,eval_batches= %d,steps_per_epoch= %d,eval_every_step= %d; eval_pct" % (
    #     real_train_pages, real_eval_pages, items_per_eval, eval_batches, steps_per_epoch, eval_every_step,eval_pct
    # ))
    # return
    # eval_every_step = int(round(items_per_eval * train_sampling / batch_size))
    # todo get_or_create_global_step 这个函数主要用于返回或者创建（如果有必要的话）一个全局步数的tensor变量。
    global_step = tf.train.get_or_create_global_step()
    # todo tf.assign_add(ref,value,use_locking=None,name=None)；通过增加value，更新ref的值，即：ref = ref + value；
    #  inc increase_step
    inc_step = tf.assign_add(global_step, 1)

    all_models: List[ModelTrainerV2] = []

    def create_model(scope, index, prefix, seed):
        # todo 主要是创建了模型，以及返回一些None的东西。
        #  数据在构建模型的时候使用了，模型中只使用了数据的预测窗口的长度--不对，应该是创建模型的时候直接喂入数据了。
        with tf.variable_scope('input') as inp_scope:
            with tf.device("/cpu:0"):
                split = splitter.splits[index]
                pipe = InputPipe(inp, features=split.train_set, n_pages=split.train_size,
                                 mode=ModelMode.TRAIN, batch_size=batch_size, n_epoch=None, verbose=verbose,
                                 train_completeness_threshold=train_completeness_threshold,
                                 predict_completeness_threshold=train_completeness_threshold, train_window=train_window,
                                 predict_window=predict_window,
                                 rand_seed=seed, train_skip_first=hparams.train_skip_first,
                                 back_offset=predict_window if forward_split else 0)
                inp_scope.reuse_variables()
                # todo side_split: False; forward_split:False; eval_stages: [];
                if side_split:
                    side_eval_pipe = InputPipe(inp, features=split.test_set, n_pages=split.test_size,
                                               mode=ModelMode.EVAL, batch_size=eval_batch_size, n_epoch=None,
                                               verbose=verbose, predict_window=predict_window,
                                               train_completeness_threshold=0.01, predict_completeness_threshold=0,
                                               train_window=train_window, rand_seed=seed, runs_in_burst=eval_batches,
                                               back_offset=predict_window * (2 if forward_split else 1))
                else:
                    side_eval_pipe = None
                if forward_split:
                    forward_eval_pipe = InputPipe(inp, features=split.test_set, n_pages=split.test_size,
                                                  mode=ModelMode.EVAL, batch_size=eval_batch_size, n_epoch=None,
                                                  verbose=verbose, predict_window=predict_window,
                                                  train_completeness_threshold=0.01, predict_completeness_threshold=0,
                                                  train_window=train_window, rand_seed=seed, runs_in_burst=eval_batches,
                                                  back_offset=predict_window)
                else:
                    forward_eval_pipe = None
        avg_sgd = asgd_decay is not None
        #asgd_decay = 0.99 if avg_sgd else None
        train_model = Model(pipe, hparams, is_train=True, graph_prefix=prefix, asgd_decay=asgd_decay, seed=seed)
        scope.reuse_variables()

        eval_stages = []
        if side_split:
            # print('2 side_split side_eval_model')
            side_eval_model = Model(side_eval_pipe, hparams, is_train=False,
                                    #loss_mask=np.concatenate([np.zeros(50, dtype=np.float32), np.ones(10, dtype=np.float32)]),
                                    seed=seed)
            # print("2  side_eval_model -- 2")
            # todo TRAIN = 0; EVAL_SIDE = 1; EVAL_FRWD = 2; EVAL_SIDE_EMA = 3; EVAL_FRWD_EMA = 4
            eval_stages.append((Stage.EVAL_SIDE, side_eval_model))
            if avg_sgd:
                eval_stages.append((Stage.EVAL_SIDE_EMA, side_eval_model))
        if forward_split:
            # print("3 forward_split forward_eval_model")
            # tf.reset_default_graph()
            forward_eval_model = Model(forward_eval_pipe, hparams, is_train=False, seed=seed)
            # print("3 forward_split forward_eval_model -- 2")
            eval_stages.append((Stage.EVAL_FRWD, forward_eval_model))
            if avg_sgd:
                eval_stages.append((Stage.EVAL_FRWD_EMA, forward_eval_model))

        if write_summaries:
            summ_path = f"{logdir}/{name}_{index}"
            # print("write_summaries summ_path",summ_path)
            if os.path.exists(summ_path):
                shutil.rmtree(summ_path)
            summ_writer = tf.summary.FileWriter(summ_path)  # , graph=tf.get_default_graph()
        else:
            summ_writer = None
        if do_eval and forward_split:
            stop_metric = lambda metrics: metrics[Stage.EVAL_FRWD]['SMAPE'].avg_epoch
        else:
            stop_metric = None
        # todo side_split: False; forward_split:False;
        #  summ_writer=<tensorflow.python.summary.writer.writer.FileWriter object at 0x7ff5dc176710>；
        #  eval_stages: []; stop_metric=None; patience=2; index=0
        # print(f"side_split: {side_split}; forward_split:{forward_split}; summ_writer={summ_writer}；"
        #       f"eval_stages: {eval_stages}； stop_metric={stop_metric}; patience={patience}; index={index}")
        return ModelTrainerV2(train_model, eval_stages, index, patience=patience,
                          stop_metric=stop_metric,
                          summary_writer=summ_writer)
    # todo n_models == 3
    if n_models == 1:
        with tf.device(f"/gpu:{gpu}"):
            scope = tf.get_variable_scope()
            all_models = [create_model(scope, 0, None, seed=seed)]
    else:
        for i in range(n_models):
            device = f"/gpu:{i}" if multi_gpu else f"/gpu:{gpu}"
            with tf.device(device):
                prefix = f"m_{i}"
                with tf.variable_scope(prefix) as scope:
                    all_models.append(create_model(scope, i, prefix=prefix, seed=seed + i))
    # todo inc_step = tf.assign_add(global_step, 1)
    trainer = MultiModelTrainer(all_models, inc_step)
    # return
    # todo save_best_model or save_from_step: False 10500
    # print("save_best_model or save_from_step: ", save_best_model, save_from_step)
    if save_best_model or save_from_step:
        saver_path = f'data/cpt/{name}'
        # todo saver_path: data/cpt/s32
        # print("saver_path: ",saver_path)
        if os.path.exists(saver_path):
            shutil.rmtree(saver_path)
        os.makedirs(saver_path)
        # todo  max_to_keep 参数，这个是用来设置保存模型的个数，默认为5，即 max_to_keep=5，保存最近的5个模型
        saver = tf.train.Saver(max_to_keep=10, name='train_saver')
    else:
        saver = None
    # todo EMA decay for averaged SGD. Not use ASGD if not set
    avg_sgd = asgd_decay is not None
    # todo asgd_decay=0.99; avg_sgd=True
    # print(f"asgd_decay={asgd_decay}; avg_sgd={avg_sgd}")
    if avg_sgd:
        from itertools import chain
        def ema_vars(model):
            ema = model.train_model.ema
            # todo: average_name() methods give access to the shadow variables and their names
            return {ema.average_name(v):v for v in model.train_model.ema._averages}
        ema_names = dict(chain(*[ema_vars(model).items() for model in all_models]))
        # todo ema_names=
        #  {'m_0/m_0/cudnn_gru/opaque_kernel/ExponentialMovingAverage': <tf.Variable 'm_0/cudnn_gru/opaque_kernel:0' shape=<unknown> dtype=float32_ref>,
        #  'm_0/m_0/fingerpint/convnet/conv1d/kernel/ExponentialMovingAverage': <tf.Variable 'm_0/fingerpint/convnet/conv1d/kernel:0' shape=(7, 5, 16) dtype=float32_ref>,
        #  'm_0/m_0/fingerpint/convnet/conv1d/bias/ExponentialMovingAverage': <tf.Variable 'm_0/fingerpint/convnet/conv1d/bias:0' shape=(16,) dtype=float32_ref>,
        #  'm_0/m_0/fingerpint/convnet/conv1d_1/kernel/ExponentialMovingAverage': <tf.Variable 'm_0/fingerpint/convnet/conv1d_1/kernel:0' shape=(3, 16, 16) dtype=float32_ref>,
        #  'm_0/m_0/fingerpint/convnet/conv1d_1/bias/ExponentialMovingAverage': <tf.Variable 'm_0/fingerpint/convnet/conv1d_1/bias:0' shape=(16,) dtype=float32_ref>,
        #  'm_0/m_0/fingerpint/convnet/conv1d_2/kernel/ExponentialMovingAverage': <tf.Variable 'm_0/fingerpint/convnet/conv1d_2/kernel:0' shape=(3, 16, 32) dtype=float32_ref>,
        #  'm_0/m_0/fingerpint/convnet/conv1d_2/bias/ExponentialMovingAverage': <tf.Variable 'm_0/fingerpint/convnet/conv1d_2/bias:0' shape=(32,) dtype=float32_ref>,
        #  'm_0/m_0/fingerpint/convnet/conv1d_3/kernel/ExponentialMovingAverage': <tf.Variable 'm_0/fingerpint/convnet/conv1d_3/kernel:0' shape=(3, 32, 32) dtype=float32_ref>,
        #  'm_0/m_0/fingerpint/convnet/conv1d_3/bias/ExponentialMovingAverage': <tf.Variable 'm_0/fingerpint/convnet/conv1d_3/bias:0' shape=(32,) dtype=float32_ref>,
        #  'm_0/m_0/fingerpint/convnet/conv1d_4/kernel/ExponentialMovingAverage': <tf.Variable 'm_0/fingerpint/convnet/conv1d_4/kernel:0' shape=(3, 32, 64) dtype=float32_ref>,
        #  'm_0/m_0/fingerpint/convnet/conv1d_4/bias/ExponentialMovingAverage': <tf.Variable 'm_0/fingerpint/convnet/conv1d_4/bias:0' shape=(64,) dtype=float32_ref>,
        #  'm_0/m_0/fingerpint/convnet/conv1d_5/kernel/ExponentialMovingAverage': <tf.Variable 'm_0/fingerpint/convnet/conv1d_5/kernel:0' shape=(3, 64, 64) dtype=float32_ref>,
        #  'm_0/m_0/fingerpint/convnet/conv1d_5/bias/ExponentialMovingAverage': <tf.Variable 'm_0/fingerpint/convnet/conv1d_5/bias:0' shape=(64,) dtype=float32_ref>,
        #  'm_0/m_0/fingerpint/fc_convnet/fc_encoder/kernel/ExponentialMovingAverage': <tf.Variable 'm_0/fingerpint/fc_convnet/fc_encoder/kernel:0' shape=(2304, 512) dtype=float32_ref>,
        #  'm_0/m_0/fingerpint/fc_convnet/fc_encoder/bias/ExponentialMovingAverage': <tf.Variable 'm_0/fingerpint/fc_convnet/fc_encoder/bias:0' shape=(512,) dtype=float32_ref>,
        #  'm_0/m_0/fingerpint/fc_convnet/out_encoder/kernel/ExponentialMovingAverage': <tf.Variable 'm_0/fingerpint/fc_convnet/out_encoder/kernel:0' shape=(512, 16) dtype=float32_ref>,
        #  'm_0/m_0/fingerpint/fc_convnet/out_encoder/bias/ExponentialMovingAverage': <tf.Variable 'm_0/fingerpint/fc_convnet/out_encoder/bias:0' shape=(16,) dtype=float32_ref>,
        #  'm_0/m_0/attn_focus/kernel/ExponentialMovingAverage': <tf.Variable 'm_0/attn_focus/kernel:0' shape=(16, 221) dtype=float32_ref>,
        #  'm_0/m_0/attn_focus/bias/ExponentialMovingAverage': <tf.Variable 'm_0/attn_focus/bias:0' shape=(221,) dtype=float32_ref>,
        #  'm_0/m_0/gru_cell/w_ru/ExponentialMovingAverage': <tf.Variable 'm_0/gru_cell/w_ru:0' shape=(291, 534) dtype=float32_ref>,
        #  'm_0/m_0/gru_cell/b_ru/ExponentialMovingAverage': <tf.Variable 'm_0/gru_cell/b_ru:0' shape=(534,) dtype=float32_ref>,
        #  'm_0/m_0/gru_cell/w_c/ExponentialMovingAverage': <tf.Variable 'm_0/gru_cell/w_c:0' shape=(291, 267) dtype=float32_ref>,
        #  'm_0/m_0/gru_cell/b_c/ExponentialMovingAverage': <tf.Variable 'm_0/gru_cell/b_c:0' shape=(267,) dtype=float32_ref>,
        #  'm_0/m_0/decoder_output_proj/kernel/ExponentialMovingAverage': <tf.Variable 'm_0/decoder_output_proj/kernel:0' shape=(267, 1) dtype=float32_ref>,
        #  'm_0/m_0/decoder_output_proj/bias/ExponentialMovingAverage': <tf.Variable 'm_0/decoder_output_proj/bias:0' shape=(1,) dtype=float32_ref>,
        # print(f"ema_names={ema_names}")
        # todo chain=<itertools.chain object at 0x7fe6587cbf98>,
        #  [] = [dict_items([
        #  ('m_0/m_0/cudnn_gru/opaque_kernel/ExponentialMovingAverage', <tf.Variable 'm_0/cudnn_gru/opaque_kernel:0' shape=<unknown> dtype=float32_ref>),
        # 	...
        #  ('m_2/m_2/decoder_output_proj/bias/ExponentialMovingAverage', <tf.Variable 'm_2/decoder_output_proj/bias:0' shape=(1,) dtype=float32_ref>)
        #  ])]
        # print(f"chain={chain(*[ema_vars(model).items() for model in all_models])},\n[] = {[ema_vars(model).items() for model in all_models]}")
        #ema_names = all_models[0].train_model.ema.variables_to_restore()
        ema_loader = tf.train.Saver(var_list=ema_names,  max_to_keep=1, name='ema_loader')
        ema_saver = tf.train.Saver(max_to_keep=1, name='ema_saver')
    else:
        ema_loader = None

    init = tf.global_variables_initializer()

    # print(f"forward_split={forward_split}; do_eval={do_eval}; side_split={side_split}")
    if forward_split and do_eval:
        eval_smape = trainer.metric(Stage.EVAL_FRWD, 'SMAPE')
        eval_mae = trainer.metric(Stage.EVAL_FRWD, 'MAE')
    else:
        eval_smape = DummyMetric()
        eval_mae = DummyMetric()

    if side_split and do_eval:
        eval_mae_side = trainer.metric(Stage.EVAL_SIDE, 'MAE')
        eval_smape_side = trainer.metric(Stage.EVAL_SIDE, 'SMAPE')
    else:
        eval_mae_side = DummyMetric()
        eval_smape_side = DummyMetric()

    train_smape = trainer.metric(Stage.TRAIN, 'SMAPE')
    train_mae = trainer.metric(Stage.TRAIN, 'MAE')
    grad_norm = trainer.metric(Stage.TRAIN, 'GrNorm')
    eval_stages = []
    ema_eval_stages = []
    if forward_split and do_eval:
        eval_stages.append(Stage.EVAL_FRWD)
        ema_eval_stages.append(Stage.EVAL_FRWD_EMA)
    if side_split and do_eval:
        eval_stages.append(Stage.EVAL_SIDE)
        ema_eval_stages.append(Stage.EVAL_SIDE_EMA)
    # todo eval_stages=[]; ema_eval_stages=[]
    # print(f"eval_stages={eval_stages}; ema_eval_stages={ema_eval_stages}")

    # gpu_options=tf.GPUOptions(allow_growth=False),
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                          gpu_options=tf.GPUOptions(allow_growth=gpu_allow_growth))) as sess:
        sess.run(init)
        # pipe.load_vars(sess)
        # todo 之前inp是加载了这个数据对象，restore是把数据tensor加载到sess中吧？
        #  这里加载了数据在哪里用到了呢？
        inp.restore(sess)
        for model in all_models:
            # todo 这里是什么意思呢？这样的到什么呢？运行了一下init_iterator？
            #  上面建好模型结构之后，在哪里喂入数据呢？
            model.init(sess)
        # if beholder:
        #    visualizer = Beholder(session=sess, logdir=summ_path)
        step = 0
        prev_top = np.inf
        best_smape = np.inf
        # Contains best value (first item) and subsequent values
        best_epoch_smape = []

        for epoch in range(max_epoch):

            # n_steps = pusher.n_pages // batch_size
            if tqdm:
                # todo Tqdm 是一个快速，可扩展的Python进度条，可以在 Python 长循环中添加一个进度提示信息，
                #  用户只需要封装任意的迭代器 tqdm(iterator)。trange(i) 是 tqdm(range(i)) 的简单写法
                #  desc=进度条前面的描述；leave：保留进度条存在的痕迹，简单来说就是会把进度条的最终形态保留下来，默认为True
                tqr = trange(steps_per_epoch, desc="%2d" % (epoch + 1), leave=False)
            else:
                tqr = range(steps_per_epoch)
            for _ in tqr:
                try:
                    # print("PRINT step = trainer.train_step")
                    # todo 训练模型只有这一行对吧
                    step = trainer.train_step(sess, epoch)
                    # if epoch == 0:
                    #     print(f"step={step}, _={_}, epoch = {epoch}")
                except tf.errors.OutOfRangeError:
                    break
                    # if beholder:
                    #  if step % 5 == 0:
                    # noinspection PyUnboundLocalVariable
                    #  visualizer.update()
                # todo 应该是每训练一个epoch，会对其中的100（eval_pct）个batch的结果做一个评估；eval_every_step= 113
                if step % eval_every_step == 0:
                    # todo eval_stages=[]；save_best_model=False; save_from_step=10500; avg_sgd=True; ema_eval_stages=[]
                    # print(f"eval_stages={eval_stages}；save_best_model={save_best_model}; save_from_step={save_from_step}; avg_sgd={avg_sgd}; ema_eval_stages={ema_eval_stages}")
                    if eval_stages:
                        trainer.eval_step(sess, epoch, step, eval_batches, stages=eval_stages)
                    if save_best_model and epoch > 0 and eval_smape.last < best_smape:
                        best_smape = eval_smape.last
                        saver.save(sess, f'data/cpt/{name}/cpt', global_step=step)
                    if save_from_step and step >= save_from_step:
                        saver.save(sess, f'data/cpt/{name}/cpt', global_step=step)

                    if avg_sgd and ema_eval_stages:
                        ema_saver.save(sess, 'data/cpt_tmp/ema',  write_meta_graph=False)
                        # restore ema-backed vars
                        ema_loader.restore(sess, 'data/cpt_tmp/ema')

                        trainer.eval_step(sess, epoch, step, eval_batches, stages=ema_eval_stages)
                        # restore normal vars
                        ema_saver.restore(sess, 'data/cpt_tmp/ema')

                MAE = "%.3f/%.3f/%.3f" % (eval_mae.last, eval_mae_side.last, train_mae.last)
                improvement = '↑' if eval_smape.improved else ' '
                SMAPE = "%s%.3f/%.3f/%.3f" % (improvement, eval_smape.last, eval_smape_side.last,  train_smape.last)
                if tqdm:
                    # todo .set_description("GEN %i"%i)	#设置进度条左边显示的信息
                    #  .set_postfix(loss=random(),gen=randint(1,999),str="h",lst=[1,2])	#设置进度条右边显示的信息
                    tqr.set_postfix(gr=grad_norm.last, MAE=MAE, SMAPE=SMAPE)
                if not trainer.has_active() or (max_steps and step > max_steps):
                    break
            if tqdm:
                tqr.close()
            trainer.end_epoch()

            if not best_epoch_smape or eval_smape.avg_epoch < best_epoch_smape[0]:
                best_epoch_smape = [eval_smape.avg_epoch]
            else:
                best_epoch_smape.append(eval_smape.avg_epoch)

            current_top = eval_smape.top
            if prev_top > current_top:
                prev_top = current_top
                has_best_indicator = '↑'
            else:
                has_best_indicator = ' '
            status = "%2d: Best top SMAPE=%.3f%s (%s)" % (
                epoch + 1, current_top, has_best_indicator,
                ",".join(["%.3f" % m.top for m in eval_smape.metrics]))

            if trainer.has_active():
                status += ", frwd/side best MAE=%.3f/%.3f, SMAPE=%.3f/%.3f; avg MAE=%.3f/%.3f, SMAPE=%.3f/%.3f, %d am" % \
                          (eval_mae.best_epoch, eval_mae_side.best_epoch, eval_smape.best_epoch, eval_smape_side.best_epoch,
                           eval_mae.avg_epoch,  eval_mae_side.avg_epoch,  eval_smape.avg_epoch,  eval_smape_side.avg_epoch,
                           trainer.has_active())
            else:
                print("Early stopping!", file=sys.stderr)
                break
            if max_steps and step > max_steps:
                print("Max steps calculated", file=sys.stderr)
                break
            sys.stderr.flush()
            # todo best_epoch_smape=[nan]; eval_smape.avg_epoch=nan; trainer.has_active()=3; prev_top=inf; current_top=nan
            # print(f"best_epoch_smape={best_epoch_smape}; eval_smape.avg_epoch={eval_smape.avg_epoch}; "
            #       f"trainer.has_active()={trainer.has_active()}; prev_top={prev_top}; current_top={current_top}")
        # noinspection PyUnboundLocalVariable
        return np.mean(best_epoch_smape, dtype=np.float64)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train the model')
    parser.add_argument('--name', default='s32', help='Model name to identify different logs/checkpoints')
    parser.add_argument('--hparam_set', default='s32', help="Hyperparameters set to use (see hparams.py for available sets)")
    parser.add_argument('--n_models', default=1, type=int, help="Jointly train n models with different seeds")
    parser.add_argument('--multi_gpu', default=False,  action='store_true', help="Use multiple GPUs for multi-model training, one GPU per model")
    parser.add_argument('--seed', default=5, type=int, help="Random seed")
    parser.add_argument('--logdir', default='data/logs', help="Directory for summary logs")
    parser.add_argument('--max_epoch', type=int, default=100, help="Max number of epochs")
    parser.add_argument('--patience', type=int, default=2, help="Early stopping: stop after N epochs without improvement. Requires do_eval=True")
    parser.add_argument('--train_sampling', type=float, default=1.0, help="Sample this percent of data for training")
    parser.add_argument('--eval_sampling', type=float, default=1.0, help="Sample this percent of data for evaluation")
    parser.add_argument('--eval_memsize', type=int, default=5, help="Approximate amount of avalable memory on GPU, used for calculation of optimal evaluation batch size")
    parser.add_argument('--gpu', default=0, type=int, help='GPU instance to use')
    parser.add_argument('--gpu_allow_growth', default=False,  action='store_true', help='Allow to gradually increase GPU memory usage instead of grabbing all available memory at start')
    parser.add_argument('--save_best_model', default=False,  action='store_true', help='Save best model during training. Requires do_eval=True')
    parser.add_argument('--no_forward_split', default=True, dest='forward_split',  action='store_false', help='Use walk-forward split for model evaluation. Requires do_eval=True')
    parser.add_argument('--side_split', default=False, action='store_true', help='Use side split for model evaluation. Requires do_eval=True')
    parser.add_argument('--no_eval', default=True, dest='do_eval', action='store_false', help="Don't evaluate model quality during training")
    parser.add_argument('--no_summaries', default=True, dest='write_summaries', action='store_false', help="Don't Write Tensorflow summaries")
    parser.add_argument('--verbose', default=False, action='store_true', help='Print additional information during graph construction')
    parser.add_argument('--asgd_decay', type=float,  help="EMA decay for averaged SGD. Not use ASGD if not set")
    parser.add_argument('--no_tqdm', default=True, dest='tqdm', action='store_false', help="Don't use tqdm for status display during training")
    parser.add_argument('--max_steps', type=int, help="Stop training after max steps")
    parser.add_argument('--save_from_step', type=int, help="Save model on each evaluation (10 evals per epoch), starting from this step")
    parser.add_argument('--predict_window', default=63, type=int, help="Number of days to predict")
    args = parser.parse_args()

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if not gpus:
        print("No GPU available")
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)


    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    config = tf.ConfigProto(allow_soft_placement=True)
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
    config.gpu_options.allow_growth = True
    # sess0 = tf.InteractiveSession(config=config)
    sess0 = tf.compat.v1.Session(config=config)

    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
    # sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))


    param_dict = dict(vars(args))
    param_dict['hparams'] = build_from_set(args.hparam_set)
    # print("param_dict ",param_dict)
    # todo 为什么这里要把这个参数删掉？删掉以后还可以用这个参数吗
    del param_dict['hparam_set']
    # print("param_dict ",param_dict)
    train(**param_dict)

    print("training over")

    # hparams = build_hparams()
    # result = train("definc_attn", hparams, n_models=1, train_sampling=1.0, eval_sampling=1.0, patience=5, multi_gpu=True,
    #                save_best_model=False, gpu=0, eval_memsize=15, seed=5, verbose=True, forward_split=False,
    #                write_summaries=True, side_split=True, do_eval=False, predict_window=63, asgd_decay=None, max_steps=11500,
    #                save_from_step=10500)

    # print("Training result:", result)
    # preds = predict('data/cpt/fair_365-15428', 380, hparams, verbose=True, back_offset=60, n_models=3)
    # print(preds)


# -*-coding:utf-8-*-

import tensorflow as tf
import os
import json
from datetime import date,timedelta
import random
import glob
import shutil


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer("dist_mode",0,'分布式模式，0：代表本地，1：单一磁盘，2：多个磁盘')
tf.app.flags.DEFINE_string("ps_hosts",'','以逗号分割的 主机:端口号 列表')
tf.app.flags.DEFINE_string('worker_hosts','','以逗号分割的 主机:端口号 列表')
tf.app.flags.DEFINE_string('job_name','','ps或worker中的一个')
tf.app.flags.DEFINE_integer('task_index',0,'job 中 task 的索引')
tf.app.flags.DEFINE_integer('num_threads',8,'线程数')
tf.app.flags.DEFINE_integer('feature_size',117581,'one hot后特征维度')
tf.app.flags.DEFINE_integer('field_size',39,'特征 field 的个数')
tf.app.flags.DEFINE_integer('embedding_size',64,'离散特征编码后的维度')
tf.app.flags.DEFINE_integer('num_epoch',1,'')
tf.app.flags.DEFINE_integer('batch_size',128,'')
tf.app.flags.DEFINE_integer('log_step',1000,'每多少step保存一次模型信息')
tf.app.flags.DEFINE_float('learning_rate',0.0005,'')
tf.app.flags.DEFINE_float('l2_reg',0.001,'')
tf.app.flags.DEFINE_string('loss_type','log_loss','损失函数的类型取下面两种类型之一{square_loss,log_loss}')
tf.app.flags.DEFINE_string('optimizer','Adam','优化器类型{Adam,Adagrad,GD,Momentum}')
tf.app.flags.DEFINE_string('deep_layers','128,64','')
tf.app.flags.DEFINE_string('dropout','0.5,0.8,0.8','')
tf.app.flags.DEFINE_boolean('batch_norm',False,'是否使用BN层')
tf.app.flags.DEFINE_float('batch_norm_decay',0.9,'')
tf.app.flags.DEFINE_string('data_dir','../../data/criteo','')
tf.app.flags.DEFINE_string('dt_dir','','日期分区？？')
tf.app.flags.DEFINE_string('model_dir','./model_ckpt/criteo/NFM/','')
tf.app.flags.DEFINE_string('servable_model_dir','','为 tf servering 加载模型')
tf.app.flags.DEFINE_string('task_type','train','取值范围：{train,infer,eval,export}')
tf.app.flags.DEFINE_boolean('clear_existing_model',False,'是否清除现在已经存在的模型')


def input_fn(file_name,batch_size,num_epoch,perform_shuffle=False):
    print('Parsing',file_name)

    def decode_libsvm(line):
        columns = tf.string_split([line],' ')
        labels = tf.string_to_number(columns.values[0],out_type=tf.float32)
        splits = tf.string_split(columns.values[1:],':')
        id_vals = tf.reshape(splits.values,splits.dense_shape)
        id,vals = tf.split(id_vals,num_or_size_splits=2,axis=1)
        feat_ids = tf.string_to_number(id,out_type=tf.int32)
        feat_vals = tf.string_to_number(vals,out_type=tf.float32)

        return {'feat_ids':feat_ids,'feat_vals':feat_vals},labels

    dataset = tf.data.TextLineDataset(file_name).map(decode_libsvm,num_parallel_calls=10).prefetch(500000)

    if perform_shuffle:
        dataset.shuffle(buffer_size=256)

    dataset = dataset.repeat(num_epoch)
    dataset = dataset.batch(batch_size)

    iterator = dataset.make_one_shot_iterator()
    batch_feature,batch_label = iterator.get_next()

    return batch_feature,batch_label


def model_fn(features, labels, mode, params):
    print('params',params)

    field_size = params['field_size']
    embedding_size = params['embedding_size']
    feature_size = params['feature_size']
    l2_reg = params['l2_reg']
    learning_rate = params['learning_rate']
    layers = list(map(int,params['deep_layer'].split(',')))
    dropout = list(map(float,params['dropout'].split(',')))

    Global_Bias = tf.get_variable('bias',shape=[1],initializer=tf.constant_initializer(0.0))
    Feat_Bias = tf.get_variable('linear',shape=[feature_size],initializer=tf.glorot_normal_initializer())
    Feat_Emb = tf.get_variable('emb',shape=[feature_size,embedding_size],initializer=tf.glorot_normal_initializer())

    feat_ids = features['feat_ids']
    feat_ids = tf.reshape(feat_ids,shape=[-1,field_size])
    feat_vals = features['feat_vals']
    feat_vals = tf.reshape(feat_vals,shape=[-1,field_size])

    with tf.variable_scope('Linear-part'):
        feat_wgts = tf.nn.embedding_lookup(Feat_Bias,feat_ids)
        y_linear = tf.reduce_sum(tf.multiply(feat_wgts,feat_vals),1)

    with tf.variable_scope('BiInter-part'):
        embedding = tf.nn.embedding_lookup(Feat_Emb,feat_ids)
        feat_vals = tf.reshape(feat_vals,[-1,field_size,1])
        embedding = tf.multiply(embedding,feat_vals)
        sum_square_emb = tf.square(tf.reduce_sum(embedding,1))
        square_sum_emb = tf.reduce_sum(tf.square(embedding),1)
        deep_input = 0.5*tf.subtract(sum_square_emb,square_sum_emb)

    with tf.variable_scope('Deep-part'):
        if mode == tf.estimator.ModeKeys.TRAIN:
            train_phase = True
        else:
            train_phase = False

        if mode == tf.estimator.ModeKeys.TRAIN:
            deep_input = tf.nn.dropout(deep_input,keep_prob=dropout[0])

        for i in range(len(layers)):
            deep_input = tf.contrib.layers.fully_connected(inputs = deep_input,num_outputs = layers[i],
                        weights_regularizer = tf.contrib.layers.l2_regularizer(l2_reg),scope = 'mlp%d' % i)

            if FLAGS.batch_norm:
                deep_input = batch_norm_layer(deep_input,train_phase = train_phase,scope_bn = 'bn_%d' % i)

            if mode == tf.estimator.ModeKeys.TRAIN:
                deep_input = tf.nn.dropout(deep_input,keep_prob=dropout[i+1])

        y_deep = tf.contrib.layers.fully_connected(inputs = deep_input,num_outputs = 1,activation_fn = tf.identity,
                weights_regularizer = tf.contrib.layers.l2_regularizer(l2_reg),scope = 'deep_out')
        y_d = tf.reshape(y_deep,[-1])

    with tf.variable_scope('NfM-out'):
        y_bias = Global_Bias * tf.ones_like(y_d,dtype=tf.float32)
        y=y_bias + y_linear + y_d
        pred = tf.sigmoid(y)

    predictions = {'prob':pred}
    export_outputs = {tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:tf.estimator.export.PredictOutput(predictions)}
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode = mode,
                    predictions=pred,
                    export_outputs=export_outputs)

    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels,logits=y) +
                l2_reg * tf.nn.l2_loss(Feat_Bias) + l2_reg * tf.nn.l2_loss(Feat_Emb))

    eval_metric_ops = {
        'auc':tf.metrics.auc(labels,pred)
    }
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode
            ,predictions=predictions
            ,loss = loss
            ,eval_metric_ops = eval_metric_ops
        )

    if FLAGS.optimizer == 'Adam':
        opt = tf.train.AdamOptimizer(learning_rate,beta1=0.9,beta2=0.999,epsilon=1e-8)
    elif FLAGS.optimizer == 'Adagrad':
        opt = tf.train.AdagradOptimizer(learning_rate,initial_accumulator_value=1e-8)
    elif FLAGS.optimizer == 'Momentum':
        opt = tf.train.MomentumOptimizer(learning_rate,momentum=0.95)
    elif FLAGS.optimizer == 'ftrl':
        opt = tf.train.FtrlOptimizer(learning_rate)

    train_op = opt.minimize(loss,global_step=tf.train.get_global_step())

    if mode == tf.estimator.ModeKeys.TRAIN:
        return tf.estimator.EstimatorSpec(
            mode
            ,predictions = predictions
            ,loss = loss
            ,train_op=train_op
        )

def batch_norm_layer(x,train_phase,scope_bn):
    bn_train = tf.contrib.layers.batch_norm(x,decay = FLAGS.batch_norm_decay,center = True,scale = True,updates_collections = None,is_training = True,reuse = None,scope = scope_bn)
    bn_infer = tf.contrib.layers.batch_norm(x,decay = FLAGS.batch_norm_decay,center = True,scale = True,updates_collections = None,is_training = False,reuse = None,scope=scope_bn)
    z = tf.cond(tf.cast(train_phase,tf.bool),lambda:bn_train,lambda:bn_infer)
    return z

def set_dist_env():
    if FLAGS.dist_mode == 1:
        ps_hosts = FLAGS.ps_hosts.split(",")
        chief_hosts = FLAGS.chief_host.split(',')
        task_index = FLAGS.task_index
        job_name = FLAGS.job_name
        tf_config = {
            'cluster':{'chief':chief_hosts,'ps':ps_hosts}
            ,'task':{'type':job_name,'index':task_index}
        }
        os.environ['TF_CONFIG'] = json.dump(tf_config)
    elif FLAGS.dist_mode == 2:
        ps_hosts = FLAGS.ps_hosts.split(',')
        worker_hosts =FLAGS.worker_host.split(',')
        chief_host = worker_hosts[0]
        worker_hosts = worker_hosts[2:]
        task_index = FLAGS.taxk_index
        job_name = FLAGS.job_name
        if job_name == 'worker' and task_index ==0:
            job_name = 'chief'
        if job_name == 'worker' and task_index == 1:
            job_name = 'evaluator'
            task_index = 0
        if job_name == 'worker' and task_index >1:
            task_index -= 2
        tf_config = {
            'cluster':{'chief':chief_host,'worker':worker_hosts,'ps':ps_hosts}
            ,'task':{'type':job_name,'index':task_index}
        }
        os.environ['TF_CONFIG'] = json.dump(tf_config)

def main(_):
    if FLAGS.dt_dir == '':
        FLAGS.dt_dir = (date.today() + timedelta(-1)).strftime('%Y%m%d')
    FLAGS.model_dir = FLAGS.model_dir + FLAGS.dt_dir

    tr_file = glob.glob('%s/tr*libsvm' %FLAGS.data_dir)
    random.shuffle(tr_file)
    va_file = glob.glob('%s/va*libsvm' % FLAGS.data_dir)
    te_file = glob.glob('%s/te*libsvm' % FLAGS.data_dir)

    if FLAGS.clear_existing_model:
        try:
            shutil.rmtree(FLAGS.model_dir)
        except Exception as e:
            print(e,'at clear_existing_model')
        else:
            print('existing model cleaned at %s ' %FLAGS.model_dir)

    set_dist_env()

    model_params = {
        'field_size':FLAGS.field_size
        ,'feature_size':FLAGS.feature_size
        ,'embedding_size':FLAGS.embedding_size
        ,'learning_rate':FLAGS.learning_rate
        ,'l2_reg':FLAGS.l2_reg
        ,'deep_layer':FLAGS.deep_layers
        ,'dropout':FLAGS.dropout
    }

    # config = tf.estimator.RunConfig().replace(
    #     session_config = tf.ConfigProto(device_count = {'CPU':FLAGS.num_threads})
    #     ,log_step_count_steps = FLAGS.log_step
    #     ,save_summary_steps = FLAGS.log_step
    # )

    config = tf.estimator.RunConfig().replace(
        session_config = tf.ConfigProto(device_count={'CPU':FLAGS.num_threads}),
        log_step_count_steps=FLAGS.log_step, save_summary_steps=FLAGS.log_step)


    Estimator = tf.estimator.Estimator(model_fn = model_fn, model_dir=FLAGS.model_dir,params=model_params,config=config)

    if FLAGS.task_type == 'train':
        train_spec = tf.estimator.TrainSpec(input_fn=lambda :input_fn(tr_file,num_epoch=FLAGS.num_epoch,batch_size=FLAGS.batch_size))
        eval_spec = tf.estimator.EvalSpec(input_fn=lambda :input_fn(va_file,num_epoch=1,batch_size=FLAGS.batch_size),steps=None,start_delay_secs=1000,throttle_secs=1200)
        tf.estimator.train_and_evaluate(Estimator,train_spec,eval_spec)

    elif FLAGS.task_type == 'eval':
        Estimator.evaluate(input_fn=lambda :input_fn(va_file,batch_size=FLAGS.vatch_size,num_epoch=1))

    elif FLAGS.task_type == 'infer':
        preds = Estimator.predict(input_fn=lambda :input_fn(te_file,batch_size=FLAGS.batch_size,num_epoch=1),predict_keys='prob')
        with open(FLAGS.data_dir+"/pred.txt",'w') as fo:
            for prob in preds:
                fo.write('%f \n' %(prob['prob']))
    elif FLAGS.task_type == 'export':
        feature_spec = {
            'feat_ids':tf.placeholder(dtype=tf.int64,shape=[None,FLAGS.field_size],name='feat_ids')
            ,'feat_vals':tf.placeholder(dtype=tf.float32,shape=[None,FLAGS.field_size],name = 'feat_vals')
        }
    serving_input_receiver_fn = tf.estimator.export.build_raw_serving_input_receiver_fn(feature_spec)
    Estimator.export_savedmodel(FLAGS.servable_model_dir,serving_input_receiver_fn)

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()







# -*-coding:utf-8-*-

import tensorflow as tf
import json
import os
import glob
from datetime import date,timedelta
import shutil

import random



FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('dist_mode',0,'分布式模式')
tf.app.flags.DEFINE_string('ps_hosts','','主机:端口号')
tf.app.flags.DEFINE_string('woker_hosts','','主机:端口号')
tf.app.flags.DEFINE_string('job_name','','"ps"或者"worker"')

tf.app.flags.DEFINE_integer('task_index',0,'job的索引')
tf.app.flags.DEFINE_integer('num_threads',16,'线程数')
tf.app.flags.DEFINE_integer('feature_size',117581,'onehot之后特征总量')
tf.app.flags.DEFINE_integer('field_size',39,'')
tf.app.flags.DEFINE_integer('embedding_size',256,'')
tf.app.flags.DEFINE_integer('num_epochs',1,'')
tf.app.flags.DEFINE_integer('batch_size',64,'')
tf.app.flags.DEFINE_integer('log_steps',1000,'')

tf.app.flags.DEFINE_float('learning_rate',0.0005,'学习率')
tf.app.flags.DEFINE_float('l2_reg',1.0,'l2正则项系数')
tf.app.flags.DEFINE_string('loss_type','los_loss','{log_loss,square_loss}')
tf.app.flags.DEFINE_string('optimizer','Adam','{Adam,Adagrad,Momentum}')
tf.app.flags.DEFINE_string('attention_layers','256','attention 层的大小')
tf.app.flags.DEFINE_string('dropout','1.0,0.5','')

tf.app.flags.DEFINE_string('data_dir','','')
tf.app.flags.DEFINE_string('dt_dir','','')
tf.app.flags.DEFINE_string('model_dir','','')
tf.app.flags.DEFINE_string('servable_model_dir','','')
tf.app.flags.DEFINE_string('task_type','train','')
tf.app.flags.DEFINE_boolean('clear_existing_model',False,'')

def input_fn(filenames,batch_size=32,num_epochs=1,perform_shuffle = False):

    def decode_libsvm(line):
        columns = tf.string_split([line],' ')
        labels = tf.string_to_number(columns.values[0],out_type=tf.float32)
        splits = tf.string_split(columns.values[1:],':')
        id_vals = tf.reshape(splits.values,splits.dense_shape)
        feat_ids,feat_vals = tf.split(id_vals,num_or_size_splits=2,axis=1)
        feat_ids = tf.string_to_number(feat_ids,out_type=tf.int32)
        feat_vals = tf.string_to_number(feat_vals,out_type=tf.float32)
        return {'feat_ids':feat_ids,'feat_vals':feat_vals},labels

    data_set = tf.data.TextLineDataset(filenames).map(decode_libsvm,num_parallel_calls=10).prefetch(50000)

    if perform_shuffle:
        data_set = data_set.shuffle(buffer_size=256)

    data_set = data_set.repeat(num_epochs)
    data_set = data_set.batch(batch_size=batch_size)

    iterator =  data_set.make_one_shot_iterator()
    batch_feat,batch_label = iterator.get_next()

    return batch_feat,batch_label

def model_fn(features,labels,mode,params):
    field_size = params['field_size']
    feature_size = params['feature_size']
    embedding_size = params['embedding_size']
    l2_reg = params['l2_reg']
    learning_rate = params['learning_rate']
    layers = list(map(int,params['attention_layer'].split(',')))
    dropout = list(map(float,params['dropout'].split(',')))

    Global_Bias = tf.get_variable('bias',shape=[1],dtype=tf.float32)
    Feat_Bias = tf.get_variable('linear',shape=[feature_size],dtype=tf.float32)
    Feat_Emb = tf.get_variable('emb',shape=[feature_size,embedding_size],dtype=tf.float32)

    feat_ids = features['feat_ids']
    feat_ids = tf.reshape(feat_ids,[-1,field_size])
    feat_vals = features['feat_vals']
    feat_vals = tf.reshape(feat_vals,shape=[-1,field_size])

    with tf.variable_scope('Linear-part'):
        line_emb = tf.nn.embedding_lookup(Feat_Bias,feat_ids)
        y_linear = tf.reduce_sum(tf.multiply(line_emb,feat_vals),1)

    with tf.variable_scope('Cross-part'):
        embedding = tf.nn.embedding_lookup(Feat_Emb,feat_ids)
        feat_vals = tf.reshape(embedding,[-1,field_size,1])
        embedding = tf.multiply(embedding,feat_vals)

        num_pair = int(field_size * (field_size - 1)/2)

        ls_pair = []
        for i in range(field_size-1):
            for j in range(i+1,field_size):
                tmp_pair = tf.multiply(embedding[:,i,:],embedding[:,j,:])
                ls_pair.append(tmp_pair)
        ls_pair = tf.transpose(tf.stack(ls_pair),[1,0,2])


    with tf.variable_scope('Attention-part'):
        deep_input = tf.reshape(ls_pair,[-1,num_pair,embedding_size])
        for i in range(len(layers)):
            deep_input = tf.contrib.layers.fully_connected(
                inputs = deep_input, num_outputs = layers[i]
                ,weights_regularizer = tf.contrib.layers.l2_regularizer(l2_reg)
                ,scope = 'mlp%d' %i
            )
        aij = tf.contrib.layers.fully_connected(
            inputs = deep_input, num_outputs = 1
            ,weights_regularizer = tf.contrib.layers.l2_regularizer(l2_reg)
            ,scope = 'attention-out'
        )

        aij = tf.reshape(aij,[-1,num_pair,1])
        aij = tf.nn.softmax(aij,dim=1,name='attention-soft')

        if mode == tf.estimator.ModeKeys.TRAIN:
            aij = tf.nn.dropout(aij,keep_prob=dropout[0])

    with tf.variable_scope('Attention-part-pooling'):
        y_emb = tf.reduce_sum(tf.multiply(ls_pair,aij),axis=1)

        if mode == tf.estimator.ModeKeys.TRAIN:
            y_emb = tf.nn.dropout(y_emb,keep_prob=dropout[1])

        y_emb = tf.contrib.layers.fully_connected(
            inputs = y_emb,num_outputs = 1
            ,weights_regularizer = tf.contrib.layers.l2_regularizer(l2_reg)
        )

        y_emb = tf.reshape(y_emb,[-1])

    with tf.variable_scope('AFM-out'):
        b = Global_Bias * tf.ones_like(y_emb,dtype=tf.float32)
        y = b + y_linear + y_emb
        pred = tf.sigmoid(y)

    predict = {'prob':pred}
    export_outputs = {tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:tf.estimator.export.PredictOutput(predict)}
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode
            ,predictions=predict
            ,export_outputs=export_outputs
        )

    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y,labels=labels)) + \
           l2_reg * tf.nn.l2_loss(Feat_Emb) + l2_reg * tf.nn.l2_loss(Feat_Bias)

    eval_metric_ops = {
        'auc':tf.metrics.auc(labels=labels,predictions=pred)
    }
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode=mode
            ,predictions=predict
            ,loss=loss
            , eval_metric_ops=eval_metric_ops
        )

    if FLAGS.optimizer == 'Adam':
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    elif FLAGS.optimizer == 'Adagrad':
        optimizer = tf.train.AdadeltaOptimizer(learning_rate=learning_rate)
    elif FLAGS.optimizer == 'Momentum':
        optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate)
    elif FLAGS.optimizer == 'ftrl':
        optimizer = tf.train.FtrlOptimizer(learning_rate=learning_rate)

    trian_op = optimizer.minimize(loss,global_step=tf.train.get_global_step())
    if mode == tf.estimator.ModeKeys.TRAIN:
        return tf.estimator.EstimatorSpec(
            mode = mode
            ,predictions=predict
            ,loss=loss
            ,train_op=trian_op
        )

def set_dist_env():
    if FLAGS.dist_mode == 1:        # 本地分布式测试模式1 chief, 1 ps, 1 evaluator
        ps_hosts = FLAGS.ps_hosts.split(',')
        chief_hosts = FLAGS.chief_hosts.split(',')
        task_index = FLAGS.task_index
        job_name = FLAGS.job_name
        print('ps_host', ps_hosts)
        print('chief_hosts', chief_hosts)
        print('job_name', job_name)
        print('task_index', str(task_index))
        # 无worker参数
        tf_config = {
            'cluster': {'chief': chief_hosts, 'ps': ps_hosts},
            'task': {'type': job_name, 'index': task_index }
        }
        print(json.dumps(tf_config))
        os.environ['TF_CONFIG'] = json.dumps(tf_config)
    elif FLAGS.dist_mode == 2:      # 集群分布式模式
        ps_hosts = FLAGS.ps_hosts.split(',')
        worker_hosts = FLAGS.worker_hosts.split(',')
        chief_hosts = worker_hosts[0:1] # get first worker as chief
        worker_hosts = worker_hosts[2:] # the rest as worker
        task_index = FLAGS.task_index
        job_name = FLAGS.job_name
        print('ps_host', ps_hosts)
        print('worker_host', worker_hosts)
        print('chief_hosts', chief_hosts)
        print('job_name', job_name)
        print('task_index', str(task_index))
        # use #worker=0 as chief
        if job_name == "worker" and task_index == 0:
            job_name = "chief"
        # use #worker=1 as evaluator
        if job_name == "worker" and task_index == 1:
            job_name = 'evaluator'
            task_index = 0
        # the others as worker
        if job_name == "worker" and task_index > 1:
            task_index -= 2

        tf_config = {
            'cluster': {'chief': chief_hosts, 'worker': worker_hosts, 'ps': ps_hosts},
            'task': {'type': job_name, 'index': task_index }
        }
        print(json.dumps(tf_config))
        os.environ['TF_CONFIG'] = json.dumps(tf_config)

def main(_):
    if FLAGS.dt_dir == '':
        FLAGS.dt_dir = (date.today()+timedelta(-1)).strftime('%Y%m%d')
    FLAGS.model_dir = FLAGS.model_dir + FLAGS.dt_dir

    tr_file = glob.glob('%s/tr*libsvm' % FLAGS.data_dir)
    te_file = glob.glob('%s/te*libsvm' % FLAGS.data_dir)
    va_file = glob.glob('%s/va*libsvm' % FLAGS.data_dir)

    if FLAGS.clear_existing_model:
        try:
            shutil.rmtree(FLAGS.clear_existing_model)
        except Exception as e:
            print(e,'at clear_existing_model')
        else:
            print('existing model clearned at ',FLAGS.model_existing_model)

    model_params = {
        'field_size' : FLAGS.field_size
        ,'feature_size':FLAGS.feature_size
        ,'embedding_size':FLAGS.embedding_size
        ,'learning_rate':FLAGS.learning_rate
        ,'l2_reg':FLAGS.l2_reg
        ,'attention_layer':FLAGS.attention_layers
        ,'dropout':FLAGS.dropout
    }

    set_dist_env()

    config = tf.estimator.RunConfig().replace(
        session_config=tf.ConfigProto(device_count={'GPU': 0, 'CPU': FLAGS.num_threads})
        ,log_step_count_steps=FLAGS.log_steps,save_summary_steps = FLAGS.log_steps
    )
    Estimator = tf.estimator.Estimator(model_fn=model_fn,model_dir=FLAGS.model_dir,config=config,params=model_params)

    if FLAGS.task_type == 'train':
        train_spec = tf.estimator.TrainSpec(input_fn=lambda :input_fn(
            tr_file, batch_size=FLAGS.batch_size, num_epochs=FLAGS.num_epochs
        ),max_steps=202)
        eval_spec = tf.estimator.EvalSpec(
            input_fn=lambda: input_fn(va_file,batch_size=FLAGS.batch_size,num_epochs=FLAGS.num_epochs)
        )
        tf.estimator.train_and_evaluate(Estimator,train_spec=train_spec,eval_spec=eval_spec)
    elif FLAGS.task_type == 'eval':
        Estimator.evaluate(input_fn=lambda :input_fn(
            va_file,batch_size=FLAGS.batch_size,num_epochs=FLAGS.num_epochs
        ))
    elif FLAGS.task_type == 'infer':
        preds = Estimator.predict(input_fn=lambda :input_fn(
            te_file,batch_size=FLAGS.batch_size,num_epochs=FLAGS.num_epochs
        ))
        with open(FLAGS.data_dir+'/pred.txt','w') as f:
            for prob in preds:
                f.write('%f\n' % (prob['prob']))

    elif FLAGS.task_type == 'export':
        feature_spec = {
            'feat_ids':tf.placeholder(dtype=tf.int64,shape=[None,FLAGS.field_size],name='feat_ids')
            ,'feat_vals':tf.placeholder(dtype=tf.float32,shape=[None,FLAGS.field],name='feat_ids')
        }
        serving_input_receiver_fn = tf.estimator.export.build_raw_serving_input_receiver_fn(features=feature_spec)
        Estimator.export_savedmodel(FLAGS.servable_model_dir,serving_input_receiver_fn=serving_input_receiver_fn)

    print('main -- over')



if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
















# 遇到一个类似18年的那个男生，每天4，5点睡觉，中午开始上班，思维很快







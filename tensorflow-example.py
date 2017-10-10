### Tensorflow example of a basic deep learning custom model function with the estimator API
## shows how to:
# -build a custom deep learning model function with the estimator api
# -use a custom decaying learning rate
# -log custom metrics during training, evaluation and prediction
# -print those metrics
#- save custom training metrics for visualization in Tensorboard

import numpy as np
import tensorflow as tf
import math
import time

tf.logging.set_verbosity(tf.logging.INFO)

model_dir = '/tensorflowlogs/example-115/'

# decaying learning rate
min_lr      = 0.00001
max_lr      = 0.01
decay_speed = 10000

# neurons per layer
L = 100
M = 60
N = 30

# optional dropout parameter for regularization, set to 1 for no dropout at all
pkeep = 0.9

# get data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# build the model
def model(features, labels, mode, params):

    # unpack 
    x     = features['x']
    pkeep = params['pkeep']
    L     = params['L']
    M     = params['M']
    N     = params['N']
    
    # weights and biases
    W1 = tf.Variable(tf.truncated_normal([784,L], stddev=0.1))
    b1 = tf.Variable(tf.ones([L])/10)
    W2 = tf.Variable(tf.truncated_normal([L,M], stddev=0.1))
    b2 = tf.Variable(tf.ones([M])/10)
    W3 = tf.Variable(tf.truncated_normal([M,N], stddev=0.1))
    b3 = tf.Variable(tf.ones([N])/10)
    W4 = tf.Variable(tf.truncated_normal([N,10], stddev=0.1))
    b4 = tf.Variable(tf.zeros(10))
    
    # layers
    y1  = tf.nn.relu(tf.matmul(x, W1) + b1)
    y1d = tf.nn.dropout(y1, pkeep)
    y2  = tf.nn.relu(tf.matmul(y1d, W2) + b2)
    y2d = tf.nn.dropout(y2, pkeep)
    y3  = tf.nn.relu(tf.matmul(y2d, W3) + b3)
    y3d = tf.nn.dropout(y3, pkeep)

    Ylogits = tf.matmul(y3d, W4) + b4
    y = tf.nn.softmax(Ylogits)
    
    # cost function
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=Ylogits)
    cost_function = tf.reduce_mean(cross_entropy)
    
    # evaluation metrics
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(labels,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')
    
    tf.summary.scalar("traincost", cost_function)

    predictions = {
        'classes': tf.argmax(input=y, axis=1),
        'accuracy': tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y,1), tf.argmax(labels,1)), tf.float32), name='accuracy_train')
    }
    
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
    
    if mode == tf.estimator.ModeKeys.TRAIN:
        
        i = tf.to_float(tf.train.get_global_step())
        lr = params['min_lr'] + (params['max_lr'] - params['min_lr']) * tf.exp(-i/params['decay_speed'])
    
        optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        train_op = optimizer.minimize(
            loss=cost_function,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=cost_function, train_op=train_op)
    
    eval_metric_ops = {
        'accuracy_test': tf.metrics.accuracy(
            labels=tf.argmax(input=labels, axis=1), predictions=predictions['classes'])         
    }
    
    return tf.estimator.EstimatorSpec(mode=mode, loss=cost_function, eval_metric_ops=eval_metric_ops)


model_params = {'min_lr': min_lr, 'max_lr': max_lr, 'decay_speed': decay_speed, 'pkeep': pkeep, 'L': L, 'M': M, 'N': N}

estimator = tf.estimator.Estimator(model_fn=model, params=model_params, model_dir=model_dir)

input_fn_train = tf.estimator.inputs.numpy_input_fn(x={'x': mnist.train.images}, y=mnist.train.labels, batch_size=100, num_epochs=None, shuffle=True)
input_fn_test  = tf.estimator.inputs.numpy_input_fn(x={'x': mnist.test.images}, y=mnist.test.labels, num_epochs=1, shuffle=False)

tensors_to_log = {'accuracy_train': 'accuracy_train'}
logging_hook = tf.train.LoggingTensorHook(
    tensors=tensors_to_log, every_n_iter=100)

summary_hook = tf.train.SummarySaverHook(
    save_steps=100,
    output_dir=model_dir,
    scaffold=tf.train.Scaffold())

# for calculation of elapsed time
t = time.time()

estimator.train(input_fn=input_fn_train, steps=10000, hooks=[logging_hook, summary_hook])

train_time = time.time() - t
print('elapsed training time: ',train_time,' seconds')

# evaluation of the model
results = estimator.evaluate(input_fn=input_fn_test, steps=None)

print("model directory = %s" % model_dir)
for key in sorted(results):
    print("%s: %s" % (key, results[key]))

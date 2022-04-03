import tensorflow as tf
import numpy as np
import cv2
# import librosa
# import librosa.display
# import matplotlib.pyplot as plt
import h5py
import os
# import scipy.io as scio
import time
# from scipy.io.wavfile import read
import soundfile as sf
from tflearn.layers.conv import global_avg_pool
time_start = time.time()


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

tf.set_random_seed(1)
learning_rate_init = 0.0001
num_examples_per_epoch_for_train = 50000
batch_size = 100
display_step = 2
drop_rate_init = 0.2
training_epochs = 500
Fs = 22050
sound_length = 29356


# file = h5py.File('mnist_data.mat')
# img_data = np.transpose(file['img_data'])
#
# sound_data = np.transpose(file['sound_data'])

data = np.load('/home/user6/project/sound_of_image/data/cifar100/cifar100_data.npz')
# img_data = data['arr_0']
sound_data = data['arr_1']


def Weights_variable(shape):
    initial = tf.truncated_normal(shape=shape, stddev=0.1, dtype=tf.float32)
    return tf.Variable(initial)


# 根据指定的维数返回初始化好的指定名称的偏置
def Biases_variable(shape):
    initial = tf.constant(0.01, shape=shape)
    return tf.Variable(initial)

def norm(inputs, on_train):
    conv_mean, conv_var = tf.nn.moments(inputs, [0])

    scale = tf.Variable(tf.ones([inputs.shape[-1]]))
    shift = tf.Variable(tf.zeros([inputs.shape[-1]]))
    epsilon = 0.001

    # apply moving average for mean and var when train on batch
    ema = tf.train.ExponentialMovingAverage(decay=0.5)

    def mean_var_with_update():
        ema_apply_op = ema.apply([conv_mean, conv_var])
        with tf.control_dependencies([ema_apply_op]):
            return tf.identity(conv_mean), tf.identity(conv_var)

    mean, var = tf.cond(on_train, mean_var_with_update, lambda: (ema.average(conv_mean), ema.average(conv_var)))

    # mean, var = mean_var_with_update()
    norm_out = tf.nn.batch_normalization(inputs, mean, var, shift, scale, epsilon)
    return norm_out

def reconstruction(sound):
    W_fc1 = Weights_variable([sound_length, 4096])
    b_fc1 = Biases_variable([4096])
    x1 = tf.nn.tanh(norm(tf.matmul(sound, W_fc1) + b_fc1, on_train))

    W_fc2 = Weights_variable([4096, 2048])
    b_fc2 = Biases_variable([2048])
    x2 = tf.nn.tanh(norm(tf.matmul(x1, W_fc2) + b_fc2, on_train))

    W_fc3 = Weights_variable([2048, 1024])
    b_fc3 = Biases_variable([1024])
    x3 = tf.nn.tanh(norm(tf.matmul(x2, W_fc3) + b_fc3, on_train))

    W_fc4 = Weights_variable([1024, 1024])
    b_fc4 = Biases_variable([1024])
    x4 = tf.nn.tanh(norm(tf.matmul(x3, W_fc4) + b_fc4, on_train))

    W_fc5 = Weights_variable([1024, 2048])
    b_fc5 = Biases_variable([2048])
    x5 = tf.nn.tanh(norm(tf.matmul(x4, W_fc5) + b_fc5, on_train))

    W_fc6 = Weights_variable([2048, 4096])
    b_fc6 = Biases_variable([4096])
    x6 = tf.nn.tanh(norm(tf.matmul(x5, W_fc6) + b_fc6, on_train))

    W_fc7 = Weights_variable([4096, sound_length])
    b_fc7 = Biases_variable([sound_length])
    x7 = tf.nn.tanh(norm(tf.matmul(x6, W_fc7) + b_fc7, on_train))
    return x7, x4

with tf.name_scope('Inputs'):
    input_placehoder = tf.placeholder(tf.float32, [batch_size, sound_length])
    # labels_input1 = mu_law_encode(labels_input, 8)
    on_train = tf.placeholder(tf.bool)
    drop_rate = tf.placeholder(dtype=tf.float32)

with tf.name_scope('Loss'):
    logits0 = reconstruction(input_placehoder)
    logits = logits0[0]
    sound_encode = logits0[1]
    # label_1024 = infer_label(labels_input)
    euclidean_loss = tf.reduce_sum(tf.square(logits - input_placehoder))

    # loss = tf.nn.softmax_cross_entropy_with_logits(
    #     logits=logits,
    #     labels=labels_input1)
    # total_loss = tf.reduce_mean(loss)
    total_loss = euclidean_loss

with tf.name_scope('Train'):
    # learning_rate = tf.placeholder(tf.float32)

    # optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate, decay=0.0005, momentum=0.9)
    optimizer = tf.train.AdamOptimizer(learning_rate_init)
    train = optimizer.minimize(total_loss)



saver = tf.train.Saver(tf.global_variables())

with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
    ckpt = tf.train.get_checkpoint_state('./stage1_model')
    saver.restore(sess, ckpt.model_checkpoint_path)
    print('===>>>>>>>==开始训练集上训练模型==<<<<<<<=====')
    total_batches = int(num_examples_per_epoch_for_train / batch_size)
    print('Per batch Size:', batch_size)
    print('Train sample Count Per Epoch:', num_examples_per_epoch_for_train)
    print('Total batch Count Per Epoch:', total_batches)
    training_step = 0
    sound_representation = np.zeros([50000, 1024])
    for batch_idx in range(total_batches):
        start_idx = batch_idx * batch_size
        end_idx = start_idx + batch_size
        batch_xs = sound_data[start_idx:end_idx, :]
        result = sess.run(sound_encode, feed_dict={input_placehoder: batch_xs,
                                             on_train: False})
        sound_representation[start_idx:end_idx, :] = result
    np.savez('/home/user6/project/sound_of_image/data/cifar100/cifar100_sound_representation.npz', sound_representation)
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
num_examples_per_epoch_for_test = 10000
batch_size = 10
display_step = 2
drop_rate_init = 0.2
Fs = 22050
sound_length = 29356
discard_rate = 0


# file = h5py.File('mnist_data.mat')
# img_data = np.transpose(file['img_data'])
#
# sound_data = np.transpose(file['sound_data'])

data = np.load('/home/user6/project/sound_of_image/data/cifar100/cifar100_test_data.npz')
img_data = data['arr_0']
sound_name = data['arr_2']




def Weights_variable(shape):
    initial = tf.truncated_normal(shape=shape, stddev=0.1, dtype=tf.float32)
    return tf.Variable(initial)


# 根据指定的维数返回初始化好的指定名称的偏置
def Biases_variable(shape):
    initial = tf.constant(0.01, shape=shape)
    return tf.Variable(initial)

def norm(inputs, on_train):
    conv_mean, conv_var = tf.nn.moments(inputs, [0, 1, 2])

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

def conv(name_str, input, w_kernel, w_bias, strides, padding, atrous_rate=None):
    with tf.name_scope(name_str):
        # Load kernel weights and apply convolution
        if not atrous_rate:
            conv_out = tf.nn.conv2d(input, w_kernel, strides, padding)
        else:
            conv_out = tf.nn.atrous_conv2d(input, w_kernel, atrous_rate, padding)
        conv_out = tf.nn.bias_add(conv_out, w_bias)
    return conv_out

def conv_layer(x, k_size, k1, k2, activation=tf.nn.relu):
    w = Weights_variable([k_size, k_size, k1, k2])
    b = Biases_variable([k2])
    x = norm(conv(name_str='pred_1', input=x,
                  w_kernel=w, w_bias=b, strides=[1, 1, 1, 1],
                  padding='SAME', atrous_rate=2), on_train)
    x = activation(x)
    return x


def sample_batch(X, Y, Z, C, size):
    start_idx = np.random.randint(0, X.shape[0]-size)
    return X[start_idx:start_idx+size, :, :], Y[start_idx:start_idx+size, :], Z[start_idx:start_idx+size, :], C[start_idx:start_idx+size, :]


def Dropout(x, rate, on_train) :
    return tf.layers.dropout(inputs=x, rate=rate, training=on_train)


def atrous_conv1d(input, dilation, k1, k2):
    w = Weights_variable([3, 1, k1, k2])
    b = Biases_variable([k2])
    out = conv(name_str='conv1', input=input, w_kernel=w, w_bias=b, strides=[1, 1, 1, 1], padding='SAME', atrous_rate=dilation)
    return out


def Inference(image):
    ########## resnet_block for image representation ###########
    trans1 = conv_layer(image, 1, 3, 64, activation=tf.nn.tanh)
    # res_block1
    x1_1 = conv_layer(trans1, 3, 64, 64, activation=tf.nn.tanh)
    x1_2 = conv_layer(x1_1, 3, 64, 64, activation=tf.identity)
    fuse1 = tf.nn.tanh(tf.add(trans1, x1_2))

    trans2 = conv_layer(fuse1, 1, 64, 128, activation=tf.nn.tanh)
    # res_block2
    x2_1 = conv_layer(trans2, 3, 128, 128, activation=tf.nn.tanh)
    x2_2 = conv_layer(x2_1, 3, 128, 128, activation=tf.identity)
    fuse2 = tf.nn.tanh(tf.add(trans2, x2_2))

    trans3 = conv_layer(fuse2, 1, 128, 256, activation=tf.nn.tanh)
    # res_block3
    x3_1 = conv_layer(trans3, 3, 256, 256, activation=tf.nn.tanh)
    x3_2 = conv_layer(x3_1, 3, 256, 256, activation=tf.identity)
    fuse3 = tf.nn.tanh(tf.add(trans3, x3_2))

    trans4 = conv_layer(fuse3, 1, 256, 512, activation=tf.nn.tanh)
    # res_block4
    x4_1 = conv_layer(trans4, 3, 512, 512, activation=tf.nn.tanh)
    x4_2 = conv_layer(x4_1, 3, 512, 512, activation=tf.identity)
    fuse4 = tf.nn.tanh(tf.add(trans4, x4_2))

    trans5 = conv_layer(fuse4, 1, 512, 1024, activation=tf.nn.tanh)
    # res_block5
    x5_1 = conv_layer(trans5, 3, 1024, 1024, activation=tf.nn.tanh)
    x5_2 = conv_layer(x5_1, 3, 1024, 1024, activation=tf.identity)
    fuse5 = tf.nn.tanh(tf.add(trans5, x5_2))

    fuse5_flat = global_avg_pool(fuse5)

    ######### fully connected layer for representatioon ########
    W_fc1 = Weights_variable([1024, 1024])
    b_fc1 = Biases_variable([1024])
    x_fc1 = tf.nn.tanh(tf.matmul(fuse5_flat, W_fc1) + b_fc1)
    x_fc1_drop = Dropout(x_fc1, discard_rate, on_train)

    W_fc1 = Weights_variable([1024, 512])
    b_fc1 = Biases_variable([512])
    x_fc1 = tf.nn.tanh(tf.matmul(x_fc1_drop, W_fc1) + b_fc1)
    x_fc1_drop = Dropout(x_fc1, discard_rate, on_train)

    ###### classification layer ########
    W_cl = Weights_variable([512, 100])
    b_cl = Biases_variable([100])
    x_class = tf.nn.softmax(tf.matmul(x_fc1_drop, W_cl) + b_cl)

    ######### fully connected layer for representatioon ########
    W_fc2 = Weights_variable([512, 1024])
    b_fc2 = Biases_variable([1024])
    x_fc2 = tf.nn.tanh(tf.matmul(x_fc1_drop, W_fc2) + b_fc2)
    x_fc2_drop = Dropout(x_fc2, discard_rate, on_train)

    W_fc2 = Weights_variable([1024, 1024])
    b_fc2 = Biases_variable([1024])
    x_fc2 = tf.nn.tanh(tf.matmul(x_fc2_drop, W_fc2) + b_fc2)
    # x_fc2_drop = Dropout(x_fc2, keep_prob, on_train)

    # W_fc3 = Weights_variable([1024, 1024])
    # b_fc3 = Biases_variable([1024])
    # x_fc3 = tf.nn.tanh(tf.matmul(x_fc2_drop, W_fc3)+b_fc3)
    #
    # x_fc3_drop = Dropout(x_fc3, keep_prob, on_train)

    x_fc2_flat = tf.reshape(x_fc2, [batch_size, 1024, 1, 1])
    #
    #
    # #########  1D atrous convolution #############
    #

    x_sigmoid_6_1 = tf.nn.sigmoid(atrous_conv1d(x_fc2_flat, 2, 1, 1))
    x_tanh_6_1 = tf.nn.tanh(atrous_conv1d(x_fc2_flat, 2, 1, 1))
    x6_1 = x_sigmoid_6_1 * x_tanh_6_1

    x6_2 = atrous_conv1d(x6_1, 2, 1, 1)
    fuse5 = tf.nn.tanh(tf.add(x_fc2_flat, x6_2))

    # trans7 = conv_layer(fuse5, 1, 16, 32, activation=tf.nn.tanh)

    x_sigmoid_7_1 = tf.nn.sigmoid(atrous_conv1d(fuse5, 2, 1, 1))
    x_tanh_7_1 = tf.nn.tanh(atrous_conv1d(fuse5, 2, 1, 1))
    x7_1 = x_sigmoid_7_1 * x_tanh_7_1

    x7_2 = atrous_conv1d(x7_1, 2, 1, 1)
    fuse7 = tf.nn.tanh(tf.add(fuse5, x7_2))

    #
    # ######## fully connected layer for output sound  #######
    #
    fuse7_flat = tf.reshape(fuse7, [batch_size, 1024])

    W_fc3 = Weights_variable([1024, 1024])
    b_fc3 = Biases_variable([1024])
    x_fc3 = tf.nn.tanh(tf.matmul(fuse7_flat, W_fc3) + b_fc3)

    x_fc3_drop = Dropout(x_fc3, discard_rate, on_train)

    W_fc4 = Weights_variable([1024, sound_length])
    b_fc4 = Biases_variable([sound_length])
    out = tf.nn.tanh(tf.matmul(x_fc3_drop, W_fc4) + b_fc4)

    return out, x_fc2, x_class


with tf.name_scope('Inputs'):
    images_input = tf.placeholder(tf.float32, [batch_size, 32, 32, 3])
    # images_input = tf.reshape(x_placeholder, [batch_size, 32, 32, 1])
    labels_input = tf.placeholder(tf.float32, [batch_size, sound_length])
    sound_encode_placeholder = tf.placeholder(tf.float32, [batch_size, 1024])
    class_label = tf.placeholder(tf.float32, [batch_size, 100])
    # labels_input1 = mu_law_encode(labels_input, 8)
    on_train = tf.placeholder(tf.bool)
    drop_rate = tf.placeholder(dtype=tf.float32)

with tf.name_scope('Loss'):
    logits0 = Inference(images_input)
    logits = logits0[0]
    sound_encode = logits0[1]
    infer_label = logits0[2]
    # label_1024 = infer_label(labels_input)

    generation_loss = tf.reduce_sum(tf.square(tf.nn.l2_normalize(logits, dim=1)
                                              - tf.nn.l2_normalize(labels_input, dim=1)))
    representation_loss = tf.reduce_sum(tf.square(tf.nn.l2_normalize(sound_encode, dim=1)
                                                  - tf.nn.l2_normalize(sound_encode_placeholder, dim=1)))

    class_loss = tf.reduce_sum(-tf.reduce_sum
    (class_label * tf.log(infer_label), reduction_indices=1))

    total_loss = generation_loss + representation_loss + class_loss

with tf.name_scope('Train'):
    # learning_rate = tf.placeholder(tf.float32)

    # optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate, decay=0.0005, momentum=0.9)
    optimizer = tf.train.AdamOptimizer(learning_rate_init)
    train = optimizer.minimize(total_loss)


initial = tf.global_variables_initializer()

saver = tf.train.Saver(tf.global_variables())


with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
    ckpt = tf.train.get_checkpoint_state('./my_model_cifar100')
    saver.restore(sess, ckpt.model_checkpoint_path)
    total_batches = int(num_examples_per_epoch_for_test / batch_size)
    print('Per batch Size:', batch_size)
    print('Train sample Count Per Epoch:', num_examples_per_epoch_for_test)
    print('Total batch Count Per Epoch:', total_batches)
    j = 0
    for epoch in range(1):
        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = start_idx + batch_size
            # batch_SR = data_SR[start_idx:end_idx]
            # batch_path = sound_path0[start_idx:end_idx]
            batch_xs = img_data[start_idx:end_idx, :, :]
            result = sess.run(logits, feed_dict={images_input: batch_xs,
                                                 on_train: False})
            for i in range(batch_size):
                sf.write(os.path.join('./test_result1/', sound_name[j] + '.wav'), result[i, :],
                         22050)
                j = j + 1
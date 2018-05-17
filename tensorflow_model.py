import tensorflow as tf
import numpy as np
import os
import re
import cv2
import h5py
import matplotlib.pyplot as plt
import time
tf.reset_default_graph()


class lazy_property(object):

    def __init__(self, func):
        self._func = func
        self.__name__ = func.__name__
        self.__doc__ = func.__doc__

    def __get__(self, obj, owner=None):
        if obj is None:
            return None
        result = obj.__dict__[self.__name__] = self._func(obj)
        return result


class Model:

    def __init__(self, data, labels, learning_rate=0.001, num_classes=None, num_epochs=None, perform_validation=False,
                 batch_size_training=32, batch_size_test=None):
        self.X_train, self.X_test = data
        self.Y_train, self.Y_test = labels
        self.learning_rate = learning_rate
        self.num_classes = num_classes
        self.num_epochs = num_epochs
        self.perform_validation = perform_validation  # True if finding accuracy on test data
        self.batch_size_training = batch_size_training
        self.batch_size_test = batch_size_test
        self.features_placeholder = tf.placeholder(dtype=tf.float32, shape=[None, 64, 64, 3])
        self.labels_placeholder = tf.placeholder(dtype=tf.float32, shape=[None, num_classes])
        self.batch_size_placeholder = tf.placeholder(dtype=tf.int64)

        # Placholder for batch norm exponential moving average Training = True, Testing = False
        self.phase = tf.placeholder(tf.bool)

        self.dataset_api()  # initialize iterator and and get_next()
        self.prediction  # Running the next 3 functions means graph in tensorflow has been made to run the model
        self.optimize
        self.cost


    @staticmethod
    # Define convolutional layer
    def conv_layer(input, channels_in, channels_out, filter_size):
        w_1 = tf.get_variable("weight_conv", [filter_size, filter_size, channels_in, channels_out],
                              initializer=tf.contrib.layers.xavier_initializer(seed=0))
        b_1 = tf.get_variable("bias_conv", [channels_out], initializer=tf.zeros_initializer())
        conv = tf.nn.conv2d(input, w_1, strides=[1, 1, 1, 1], padding="SAME")
        activation = tf.nn.relu(conv + b_1)
        activation_without_bias = tf.nn.relu(conv)
        return activation

    @staticmethod
    # Define fully connected layer
    def fc_layer(input, channels_in, channels_out):
        w_2 = tf.get_variable("weight_fc", [channels_in, channels_out],
                              initializer=tf.contrib.layers.xavier_initializer(seed=0))
        b_2 = tf.get_variable("bias_fc", [channels_out], initializer=tf.zeros_initializer())
        activation = tf.nn.relu(tf.matmul(input, w_2) + b_2)
        activation_without_bias = tf.nn.relu(tf.matmul(input, w_2))
        return activation

    @staticmethod
    # Batch normalisation of convolution layer
    def conv_batch_norm(input, num_channels, phase):
        beta = tf.get_variable("beta", shape=[num_channels], initializer=tf.zeros_initializer())
        gamma = tf.get_variable("gamma", shape=[num_channels], initializer=tf.ones_initializer())

        # Shape of batch_mean/var = [num_channels], assuming input = [batch size, height, width, channel]
        batch_mean, batch_var = tf.nn.moments(input, axes=[0, 1, 2])

        ema = tf.train.ExponentialMovingAverage(decay=0.8)

        def population_moving_average():
            ema_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_op]):
                # Have to use tf.identity to specify that batch_mean/var tensor is run only after evaluation ema_op
                return tf.identity(batch_mean), tf.identity(batch_var)

        # tf.cond returns different values depending if the first input is true or false
        # Returns population moving average if phase is 'true' (i.e. training) or exponential moving average else
        mean, variance = tf.cond(phase, population_moving_average,
                                 lambda: (ema.average(batch_mean), (ema.average(batch_var))))

        normed_output = tf.nn.batch_normalization(input, mean=mean,
                                                  variance=variance,
                                                  offset=beta, scale=gamma,
                                                  variance_epsilon=1e-8)

        return normed_output

    def dataset_api(self):
        # Define dataset api
        dataset = tf.data.Dataset.from_tensor_slices(
            (self.features_placeholder,  # Slices 0th dimension of features and labels
             self.labels_placeholder))
        dataset = dataset.batch(self.batch_size_placeholder)
        dataset = dataset.shuffle(buffer_size=1080)
        self.iterator = dataset.make_initializable_iterator()
        self.x, self.y = self.iterator.get_next()

    @lazy_property
    def prediction(self):
        # Define network
        with tf.variable_scope("conv1"):
            Z1 = self.conv_layer(self.x, 3, 8, filter_size=4)
            Z1_batch_norm = self.conv_batch_norm(Z1, num_channels=8, phase=self.phase)

        with tf.variable_scope("conv2"):
            Z2 = self.conv_layer(Z1_batch_norm, 8, 16, filter_size=4)
            Z2_batch_norm = self.conv_batch_norm(Z2, num_channels=16, phase=self.phase)
            P2 = tf.nn.max_pool(Z2_batch_norm, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1],
                                padding="VALID")  # For future reference, padding only works if strides are unit length

        with tf.variable_scope("conv3"):
            Z3 = self.conv_layer(P2, 16, 16, filter_size=4)
            Z3_batch_norm = self.conv_batch_norm(Z3, num_channels=16, phase=self.phase)

        with tf.variable_scope("conv4"):
            Z4 = self.conv_layer(Z3_batch_norm, 16, 8, filter_size=4)
            Z4_batch_norm = self.conv_batch_norm(Z4, num_channels=8, phase=self.phase)
            P4 = tf.nn.max_pool(Z4_batch_norm, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding="VALID")

        with tf.variable_scope("conv5"):
            Z5 = self.conv_layer(P4, 8, 8, filter_size=4)
            Z5_batch_norm = self.conv_batch_norm(Z5, num_channels=8, phase=self.phase)

        with tf.variable_scope("conv6"):
            Z6 = self.conv_layer(Z5_batch_norm, 8, 128, filter_size=2)
            Z6_batch_norm = self.conv_batch_norm(Z6, num_channels=128, phase=self.phase)
            P6 = tf.nn.max_pool(Z6_batch_norm, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding="VALID")

        with tf.name_scope("FC_layer1"):
            flattened = tf.contrib.layers.flatten(P6)

            # No. input channels = W*H*Channels, note W*H hasn't changed from original due to use of "SAME" padding
            Z7 = self.fc_layer(flattened, 1 * 1 * 128,self.num_classes)
        return Z7

    @lazy_property
    def cost(self):
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.prediction, labels=self.y))
        return cost

    @lazy_property
    def optimize(self):
        train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cost)
        return train_op

    def compute_accuracy(self, labels):
        predict_op = tf.argmax(self.prediction, 1)
        correct_prediction = tf.equal(predict_op, tf.argmax(labels, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        def my_tf_round(x, decimals=0):
            multiplier = tf.constant(10 ** decimals, dtype=x.dtype)
            return tf.round(x * multiplier) / multiplier

        return my_tf_round(accuracy, decimals=3)


# Change labels to one hot version
def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y


# Google_drive variable is True if running code in google colab
def load_dataset(google_drive=False):
    if google_drive is False:
        train_dataset = h5py.File('C:\\Users\\tharu\\Desktop\\PythonCode\\Tensorflow\\owo\\train_signs.h5', "r")
    else:
        train_dataset = h5py.File('drive/app/train_signs.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])  # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])  # your train set labels

    if google_drive is False:
        test_dataset = h5py.File('C:\\Users\\tharu\\Desktop\\PythonCode\\Tensorflow\\owo\\test_signs.h5', "r")
    else:
        test_dataset = h5py.File('drive/app/test_signs.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])  # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])  # your test set labels

    classes = np.array(test_dataset["list_classes"][:])  # the list of classes

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


# Load data
X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset(google_drive=False)

# Rescale images and convert labels to one hot vectors
X_train = X_train_orig/255.
X_test = X_test_orig/255.
Y_train = convert_to_one_hot(Y_train_orig, 6).T
Y_test = convert_to_one_hot(Y_test_orig, 6).T

data = X_train, X_test
labels = Y_train, Y_test

# Define parameters
learning_rate = 0.01
num_classes = 6
EPOCHS = 100
perform_validation = True
batch_size_training = 5
batch_size_test = X_test.shape[0]  # 0th dimension gives no. examples (we want to test entire test set)

model = Model(data, labels, learning_rate=0.001, num_classes=num_classes, num_epochs=EPOCHS,
              perform_validation=perform_validation, batch_size_training=batch_size_training,
              batch_size_test=batch_size_test)

iterator = model.iterator

start_time = time.time()

with tf.Session() as sess:
    # Initiliaze all the variables
    sess.run(tf.global_variables_initializer())

    # Train the network
    for epoch_num in range(EPOCHS):
        # Initialize iterator so that it starts at beginning of training set for each epoch
        sess.run(iterator.initializer,
                 feed_dict={model.features_placeholder: model.X_train,
                            model.labels_placeholder: model.Y_train,
                            model.batch_size_placeholder: model.batch_size_training})
        while True:
            try:
                _, epoch_loss = sess.run([model.optimize, model.cost], feed_dict={model.phase: True})

            except tf.errors.OutOfRangeError:  # Error given when out of data
                break

        # Get updates on how training is going
        if epoch_num % 1 == 0:
            sess.run(iterator.initializer,
                     feed_dict={model.features_placeholder: model.X_train,
                                model.labels_placeholder: model.Y_train,
                                model.batch_size_placeholder: X_train.shape[0]})
            while True:
                try:
                    print("Epoch number:", epoch_num, "\t", "Loss:", round(epoch_loss, 3), "\t" "Training accuracy:",
                          sess.run(model.compute_accuracy(model.y), feed_dict={model.phase: False}))
                except tf.errors.OutOfRangeError:
                    break

    # Run validation on test set
    if perform_validation is True:
        print("STARTING VALIDATION")
        sess.run(iterator.initializer,
                 feed_dict={model.features_placeholder: model.X_test,
                            model.labels_placeholder: model.Y_test,
                            model.batch_size_placeholder: model.batch_size_test})
        while True:
            try:
                print("Test accuracy:", sess.run(model.compute_accuracy(model.y), feed_dict={model.phase: False}))
            except tf.errors.OutOfRangeError:
                break

elapsed_time = time.time() - start_time
print("Minutes running: ", round(elapsed_time/60, 2))

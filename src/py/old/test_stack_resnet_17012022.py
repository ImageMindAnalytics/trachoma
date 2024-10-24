from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.utils import OrderedEnqueuer
from keras import backend
from keras.utils import control_flow_util
import json
import os
import glob
import sys
import pandas as pd
import SimpleITK as sitk
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from sklearn.utils import shuffle
from scipy import ndimage
import pickle

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
        

class Attention(tf.keras.layers.Layer):
    def __init__(self, units, w_units):
        super(Attention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(w_units)

    def call(self, query, values):        

        # score shape == (batch_size, max_length, 1)
        # we get 1 at the last axis because we are applying score to self.V
        # the shape of the tensor before applying self.V is (batch_size, max_length, units)
        score = self.V(tf.nn.tanh(self.W1(query)))
        
        attention_weights = tf.nn.softmax(score, axis=1)

        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, score

# class RandomIntensity(tf.keras.layers.Layer):
#     def call(self, x, training=True):
#         if training is None:
#           training = backend.learning_phase()
#         # return control_flow_util.smart_cond(training, 
#         return tf.cond(tf.cast(training, tf.bool), 
#             lambda: self.random_intensity(x),
#             lambda: x)
#     def random_intensity(self, x):
#         r = tf.random.uniform(shape=[], maxval=5, dtype=tf.int32)
#         x = tf.cond(r == 1, lambda: self.saturation(x), lambda: x)
#         x = tf.cond(r == 2, lambda: self.contrast(x), lambda: x)
#         x = tf.cond(r == 3, lambda: self.hue(x), lambda: x)
#         x = tf.cond(r == 4, lambda: self.brightness(x), lambda: x)
#         return x
#     def saturation(self, x):
#         return tf.image.random_saturation(x, 0, 10)
#     def contrast(self, x):
#         return tf.image.random_contrast(x, 0, 0.5)
#     def hue(self, x):
#         return tf.image.random_hue(x, 0.25)
#     def brightness(self, x):
#         return tf.image.random_brightness(x, 70)

class Features(tf.keras.layers.Layer):
    def __init__(self):
        super(Features, self).__init__()

        self.resnet = tf.keras.applications.ResNet50V2(include_top=False, weights='imagenet', input_tensor=tf.keras.Input(shape=[448, 448, 3]), pooling=None)

        # self.random_intensity = RandomIntensity()
        # self.random_rotation = tf.keras.layers.RandomRotation(0.5)
        # self.random_zoom = tf.keras.layers.RandomZoom(0.1, fill_mode='reflect')
        # self.random_crop = tf.keras.layers.RandomCrop(448, 448)
        # self.center_crop = tf.keras.layers.CenterCrop(448, 448)
        self.rescale = layers.Rescaling(1/127.5, offset=-1)
        self.conv = layers.Conv2D(512, (2, 2), strides=(2, 2))
        self.avg = layers.GlobalAveragePooling2D()

    def compute_output_shape(self, input_shape):
        return (None, 512)


    def call(self, x):

        # x = self.random_intensity(x)
        # x = self.random_rotation(x)
        # x = self.random_zoom(x)
        # x = self.random_crop(x)
        # x = self.center_crop(x)
        x = self.rescale(x)
        x = self.resnet(x)
        x = self.conv(x)
        x = self.avg(x)

        return x


class TTModel(tf.keras.Model):
    def __init__(self, features = None):
        super(TTModel, self).__init__()

        self.features = Features()
        if(features):
            self.features.resnet = features.resnet
            self.features.conv = features.conv

        self.TD = layers.TimeDistributed(self.features)
        self.R = layers.Reshape((-1, 512))

        self.V = layers.Dense(256)
        self.A = Attention(128, 1)        
        self.P = layers.Dense(2, activation='softmax', name='predictions')
        
    def call(self, x):

        x = self.TD(x)
        x = self.R(x)

        x_v = self.V(x)
        x_a, x_s = self.A(x, x_v)
        
        x = self.P(x_a)
        x_v_p = self.P(x_v)

        return x, x_a, x_v, x_s, x_v_p


class DatasetGenerator(tf.keras.utils.Sequence):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
            
        row = self.df.loc[idx]
        img = os.path.join("/work/jprieto/data/remote/EGower/", row["img"])
        sev = row["class"]

        img_np = sitk.GetArrayFromImage(sitk.ReadImage(img))

        t, xs, ys, _ = img_np.shape
        xo = (xs - 448)//2
        yo = (ys - 448)//2
        img_np = img_np[:,xo:xo + 448, yo:yo + 448,:]
        
        one_hot = np.zeros(2)
        one_hot[sev] = 1

        return img_np, one_hot



checkpoint_path = "/work/jprieto/data/remote/EGower/jprieto/train/stack_training_resnet_att_17012022_weights/stack_training_resnet_att_17012022"
    

model = TTModel()
model.load_weights(checkpoint_path)
model.build(input_shape=(None, None, 448, 448, 3))
model.summary()



csv_path_stacks = "/work/jprieto/data/remote/EGower/jprieto/hinashah/Analysis_Set_01132022/trachoma_normals_healthy_sev123_epi_stack_16_768_test.csv"

test_df = pd.read_csv(csv_path_stacks).replace("/work/jprieto/data/remote/EGower/", "", regex=True)
test_df['class'] = (test_df['class'] >= 1).astype(int)

dg_test = DatasetGenerator(test_df)

def test_generator_2():

    enqueuer = OrderedEnqueuer(dg_test, use_multiprocessing=True)
    enqueuer.start(workers=8, max_queue_size=128)

    datas = enqueuer.get()

    for idx in range(len(dg_test)):
        yield next(datas)

    enqueuer.stop()

dataset = tf.data.Dataset.from_generator(test_generator_2,
    output_signature=(tf.TensorSpec(shape = (None, 448, 448, 3), dtype = tf.float32), 
        tf.TensorSpec(shape = (2,), dtype = tf.int32))
    )

dataset = dataset.batch(1)
dataset = dataset.prefetch(16)



dataset_stacks_predict = model.predict(dataset, verbose=True)
output_dir = "/work/jprieto/data/remote/EGower/jprieto/test_output/"

with open(os.path.join(output_dir, 'trachoma_normals_healthy_sev123_epi_stack_16_768_test_17012022.pickle'), 'wb') as f:
    pickle.dump(dataset_stacks_predict, f)




















csv_path_stacks = "/work/jprieto/data/remote/EGower/jprieto/trachoma_normals_healthy_sev123_05182021_stack_16_544_10082021_test.csv"

test_df = pd.read_csv(csv_path_stacks).replace("/work/jprieto/data/remote/EGower/", "", regex=True)
test_df['class'] = (test_df['class'] >= 1).astype(int)


dg_test = DatasetGenerator(test_df)

def test_generator():

    enqueuer = OrderedEnqueuer(dg_test, use_multiprocessing=True)
    enqueuer.start(workers=8, max_queue_size=128)

    datas = enqueuer.get()

    for idx in range(len(dg_test)):
        yield next(datas)

    enqueuer.stop()

dataset = tf.data.Dataset.from_generator(test_generator,
    output_signature=(tf.TensorSpec(shape = (None, 448, 448, 3), dtype = tf.float32), 
        tf.TensorSpec(shape = (2,), dtype = tf.int32))
    )

dataset = dataset.batch(1)
dataset = dataset.prefetch(16)


dataset_stacks_predict = model.predict(dataset, verbose=True)
output_dir = "/work/jprieto/data/remote/EGower/jprieto/test_output/"

with open(os.path.join(output_dir, 'trachoma_normals_healthy_sev123_05182021_stack_16_544_10082021_test_17012022.pickle'), 'wb') as f:
    pickle.dump(dataset_stacks_predict, f)
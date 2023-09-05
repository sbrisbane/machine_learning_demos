#!/usr/bin/env python3
!nvidia-smi

import requests, gzip, pickle
def load_minst_data():
  
  open("minst.pkl.gz", "wb").write(requests.get("https://s3.amazonaws.com/img-datasets/mnist.pkl.gz").content)
  print (gzip.open("minst.pkl.gz", "rb"))
  with gzip.open("minst.pkl.gz", "rb") as f:
     return pickle.load(f,encoding='bytes')



import tensorflow as tf
#import tensorflow_datasets as tfds
from tensorflow.keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = load_minst_data()

x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(1280 , activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(1280 , activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(10 , activation=tf.nn.softmax))

model.compile (optimizer="adam", loss="sparse_categorical_crossentropy",
   metrics=["accuracy"])
model.fit(x=x_train, y=y_train,epochs=5)

#print("\nTest accuracy: " , str(test_acc*100) , "%" )

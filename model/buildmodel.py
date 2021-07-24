import tensorflow as tf

from tensorflow.keras import layers


def build_model(num_labels):
  
  model = tf.keras.Sequential([
  layers.Dense(num_labels)])

  return model

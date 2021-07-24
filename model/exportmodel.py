import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import losses

def export_model(model, vectorize_layer):
  export_model = tf.keras.Sequential([
    vectorize_layer,
    model,
    layers.Activation('sigmoid')
  ])

  export_model.compile(
      loss=losses.SparseCategoricalCrossentropy(), optimizer="adam", metrics=['accuracy']
  )

  return export_model

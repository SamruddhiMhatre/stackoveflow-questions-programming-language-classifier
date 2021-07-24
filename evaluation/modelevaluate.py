import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import losses
from model import export_model


export_model = export_model(model, vectorize_layer)

# Test
loss, accuracy = export_model.evaluate(raw_test_ds)
print(accuracy)

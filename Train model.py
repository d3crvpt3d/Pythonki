from data import get_mnist
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

###from keras website
###

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Preprocess the data (these are NumPy arrays)
x_train = x_train.reshape(60000, 784).astype("float32") / 255
x_test = x_test.reshape(10000, 784).astype("float32") / 255

y_train = y_train.astype("float32")
y_test = y_test.astype("float32")

# Reserve 10,000 samples for validation
x_val = x_train[-10000:]
y_val = y_train[-10000:]
x_train = x_train[:-10000]
y_train = y_train[:-10000]

###
###from keras website



model = keras.Sequential(
    [
        layers.Dense(784, activation="relu"),
        layers.Dense(20, activation="relu"),
        layers.Dense(20, activation="relu"),
        layers.Dense(10, activation="softmax")
    ]
)


model.compile(optimizer="adam", loss=keras.losses.SparseCategoricalCrossentropy(), metrics=[keras.metrics.SparseCategoricalAccuracy()])


model.fit(x_train, y_train, batch_size=32, epochs=2, validation_data=(x_val, y_val))


print("Evaluate on test data")
results = model.evaluate(x_test, y_test, batch_size=128)
print("test loss, test acc:", results)

model.save('model.h5')
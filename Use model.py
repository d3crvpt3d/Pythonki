from data import get_mnist
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import load_model

model = load_model('model.h5')

# A few random samples
use_samples = [5, 38, 3939, 27389]

# Convert into Numpy array
samples_to_predict = np.array(use_samples)

# Generate predictions for samples
predictions = model.predict(samples_to_predict)
print(predictions)
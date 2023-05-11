import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
import numpy as np
import lime
from lime import lime_image
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
import random
import os


# Helper functions
def to_rgb(x):
    """
    Converts a grayscale image to RGB
    """
    x_rgb = np.zeros((x.shape[0], 28, 28, 3))
    for i in range(3):
        x_rgb[..., i] = x[..., 0]
    return x_rgb

def create_and_save_model():
    """
    Creates and saves a model
    """
    # Load the MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Normalize the data
    x_train = x_train.reshape((-1, 28, 28, 1)).astype('float32') / 255.0
    x_test = x_test.reshape((-1, 28, 28, 1)).astype('float32') / 255.0

    # Convert the grayscale images to RGB
    x_train = to_rgb(x_train)
    x_test = to_rgb(x_test)

    # Create the model
    model = keras.Sequential([
        keras.Input(shape=(28, 28, 3)),
        layers.Conv2D(16, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(10)
    ])

    # Compile the model
    model.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=keras.optimizers.Adam(),
        metrics=['accuracy']
    )

    # Fit the model
    model.fit(x_train, y_train, epochs=2, batch_size=32, validation_data=(x_test, y_test))

    # Save the model
    model.save('model.h5')

    return model

def get_model():
    """
    Loads the model from the file system
    """
    # Check if the model exists
    if not os.path.exists('model.h5'):
        return create_and_save_model()
    
    return keras.models.load_model('model.h5')

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize the data
x_train = x_train.reshape((-1, 28, 28, 1)).astype('float32') / 255.0
x_test = x_test.reshape((-1, 28, 28, 1)).astype('float32') / 255.0

# Convert the grayscale images to RGB
x_train = to_rgb(x_train)
x_test = to_rgb(x_test)

model = get_model()

# Create the lime explainer
explainer = lime_image.LimeImageExplainer(random_state=42)
explanation = explainer.explain_instance(
         x_train[10], 
         model.predict
)
# plt.imshow(x_train[10])
image, mask = explanation.get_image_and_mask(
         model.predict(
              x_train[10].reshape((1,28,28,3))
         ).argmax(axis=1)[0],
         positive_only=True, 
         hide_rest=False)

top_label = explanation.top_labels[0]

fig, ax = plt.subplots()
ax.imshow(mark_boundaries(image, mask))
ax.set_title('Explanation with LIME (Label: ' + str(top_label) + ')', fontsize=12, fontweight='bold')
# Text under the image

plt.show()
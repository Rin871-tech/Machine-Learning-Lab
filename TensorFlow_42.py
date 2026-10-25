# -*- coding: utf-8 -*-
"""
TensorFlow_42.py
Beginner-to-Advanced Guide to TensorFlow
Install: python -m pip install tensorflow matplotlib numpy
Note: TensorFlow installation support depends on operating system and hardware.
"""

# %% 1. INTRODUCTION
"""
TensorFlow is an open-source platform for numerical computation and machine learning.
It supports tensors, automatic differentiation, neural networks, model training,
deployment, and acceleration on supported hardware.

A tensor is a multi-dimensional array:
- scalar: rank 0
- vector: rank 1
- matrix: rank 2
- higher-rank tensor: rank 3 or more

Keras is TensorFlow's high-level API for building neural networks.
"""

# %% 2. VERSION AND BASIC TENSORS
import tensorflow as tf
import numpy as np

print("TensorFlow version:", tf.__version__)
a = tf.constant(10)
b = tf.constant([1, 2, 3])
matrix = tf.constant([[1., 2.], [3., 4.]])
print("Scalar:", a)
print("Vector:", b)
print("Matrix:\n", matrix)
print("Shape:", matrix.shape, "dtype:", matrix.dtype)

# %% 3. TENSOR OPERATIONS
x = tf.constant([[1., 2.], [3., 4.]])
y = tf.constant([[5., 6.], [7., 8.]])
print("Addition:\n", x + y)
print("Elementwise multiplication:\n", x * y)
print("Matrix multiplication:\n", tf.matmul(x, y))
print("Reshape:", tf.reshape(tf.range(12), (3, 4)))
print("Reduction sum:", tf.reduce_sum(x).numpy())

# %% 4. VARIABLES AND UPDATES
weight = tf.Variable(2.0)
weight.assign_add(3.0)
print("Updated variable:", weight.numpy())

# %% 5. AUTOMATIC DIFFERENTIATION WITH GRADIENTTAPE
x = tf.Variable(3.0)
with tf.GradientTape() as tape:
    y = x * x + 2 * x + 1
gradient = tape.gradient(y, x)
print("f(x) =", y.numpy(), "derivative =", gradient.numpy())

# %% 6. DATA PIPELINES WITH TF.DATA
values = np.arange(20, dtype=np.float32)
dataset = tf.data.Dataset.from_tensor_slices(values)
dataset = dataset.shuffle(20, seed=42).batch(5).prefetch(tf.data.AUTOTUNE)
for batch in dataset.take(2):
    print("Batch:", batch.numpy())

# %% 7. FIRST NEURAL NETWORK: MNIST CLASSIFICATION
"""
This example downloads MNIST the first time it runs. Internet access is required.
"""
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

model = keras.Sequential([
    layers.Input(shape=(28, 28)),
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.2),
    layers.Dense(10, activation="softmax"),
])
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
model.summary()
history = model.fit(
    x_train, y_train, epochs=3, batch_size=128,
    validation_split=0.1, verbose=1
)
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
print("Test loss:", test_loss, "Test accuracy:", test_accuracy)

# %% 8. PLOT TRAINING HISTORY
plt.figure(figsize=(8, 4))
plt.plot(history.history["accuracy"], label="training accuracy")
plt.plot(history.history["val_accuracy"], label="validation accuracy")
plt.title("MNIST Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.tight_layout()
plt.show()

# %% 9. MAKE PREDICTIONS
probabilities = model.predict(x_test[:5], verbose=0)
predicted_labels = np.argmax(probabilities, axis=1)
print("Predicted:", predicted_labels)
print("Actual:   ", y_test[:5])

# %% 10. CALLBACKS: EARLY STOPPING AND CHECKPOINTING
"""
Callbacks can stop training when validation performance stops improving and
save the best model. This snippet demonstrates configuration; use it in fit().
"""
early_stop = keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=2, restore_best_weights=True
)
checkpoint = keras.callbacks.ModelCheckpoint(
    "best_mnist_model.keras", monitor="val_loss", save_best_only=True
)
# Example:
# model.fit(x_train, y_train, validation_split=0.1,
#           epochs=10, callbacks=[early_stop, checkpoint])

# %% 11. SAVE AND LOAD A MODEL
model.save("mnist_tensorflow_model.keras")
reloaded_model = keras.models.load_model("mnist_tensorflow_model.keras")
print("Reloaded model test accuracy:",
      reloaded_model.evaluate(x_test, y_test, verbose=0)[1])

# %% 12. CNN FOR IMAGE CLASSIFICATION
"""
Convolutional neural networks are useful for image-like inputs. This example
uses a small CNN on MNIST and is kept separate from the dense model above.
"""
cnn = keras.Sequential([
    layers.Input(shape=(28, 28, 1)),
    layers.Rescaling(1.0 / 255),
    layers.Conv2D(32, 3, activation="relu"),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation="relu"),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(64, activation="relu"),
    layers.Dense(10, activation="softmax"),
])
cnn.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
# Reshape raw pixel arrays to add the channel dimension.
# Train for one epoch as a quick demonstration; increase epochs for experiments.
cnn.fit(x_train[..., np.newaxis] * 255.0, y_train,
        epochs=1, batch_size=128, validation_split=0.1)
print("CNN test metrics:", cnn.evaluate(x_test[..., np.newaxis] * 255.0, y_test, verbose=0))

# %% 13. CUSTOM TRAINING LOOP
"""
For advanced workflows, GradientTape lets you write a training loop manually.
This minimal example optimizes y = 3x + 2 using a linear model.
"""
train_x = tf.constant(np.linspace(-1, 1, 200).astype("float32"))
train_y = 3 * train_x + 2
w = tf.Variable(0.0)
bias = tf.Variable(0.0)
optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)

for epoch in range(100):
    with tf.GradientTape() as tape:
        predictions = w * train_x + bias
        loss = tf.reduce_mean(tf.square(predictions - train_y))
    gradients = tape.gradient(loss, [w, bias])
    optimizer.apply_gradients(zip(gradients, [w, bias]))
print("Learned w:", w.numpy(), "bias:", bias.numpy(), "loss:", loss.numpy())

# %% 14. GPU AVAILABILITY
print("Available GPU devices:", tf.config.list_physical_devices("GPU"))
"""
A GPU is optional. TensorFlow may use CPU if no supported GPU is available.
Hardware setup differs across Windows, macOS, and Linux.
"""

# %% 15. COMMON LAYERS AND LOSSES
"""
Layers:
- Dense: fully connected layer
- Conv2D: 2D convolution for images
- MaxPooling2D: downsamples feature maps
- Embedding: maps integer IDs to vectors
- LSTM / GRU: recurrent layers for sequence data
- Dropout: regularization during training
- BatchNormalization: stabilizes/accelerates training in many settings

Losses:
- sparse_categorical_crossentropy: integer class labels
- categorical_crossentropy: one-hot class labels
- binary_crossentropy: binary/multi-label outputs
- mean_squared_error: regression
"""

# %% 16. BEST PRACTICES AND TROUBLESHOOTING
"""
- Normalize inputs using training-appropriate preprocessing.
- Keep training, validation, and test data separate.
- Match the output layer and loss to the target format.
- Start with a small model and baseline.
- Monitor validation metrics to spot overfitting.
- Set seeds when reproducibility matters, but exact results may still vary by hardware.
- Use model.save() with the .keras format for whole-model saving.
- Do not expect a GPU to be available on every machine.
"""

# %% 17. PRACTICE EXERCISES
"""
1. Change the hidden-layer width and compare validation accuracy.
2. Add EarlyStopping to the MNIST training.
3. Try a CNN with a different number of filters.
4. Train a regression model on synthetic data.
5. Create a tf.data pipeline that shuffles, batches, and prefetches.
6. Compare SGD and Adam.
7. Save a model and use it in a separate prediction script.
"""
print("TensorFlow guide finished.")

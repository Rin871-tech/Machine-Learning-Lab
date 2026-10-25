# -*- coding: utf-8 -*-
"""
Keras_42.py
Complete Learning Guide to Keras for Neural Networks

Keras is a high-level deep-learning API. In TensorFlow projects, use:
    from tensorflow import keras
    from tensorflow.keras import layers

Install: python -m pip install tensorflow numpy matplotlib
Keras 3 can also run with other supported backends, but this guide uses TensorFlow.
"""

# %% 1. INTRODUCTION
"""
Keras helps you build, train, evaluate, and save neural networks with a readable API.

Typical workflow:
1. Prepare X and y.
2. Define a model.
3. Compile with an optimizer, loss, and metrics.
4. Train with fit().
5. Evaluate on held-out data.
6. Predict and save the model.

Model styles:
- Sequential: simple stack of layers.
- Functional API: branching, multiple inputs/outputs, shared layers.
- Model subclassing: custom forward-pass behavior.
"""

# %% 2. IMPORTS AND VERSION
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt

print("TensorFlow:", tf.__version__)
print("Keras API available:", hasattr(keras, "Sequential"))

# %% 3. A SIMPLE SEQUENTIAL MODEL
model = keras.Sequential([
    layers.Input(shape=(4,)),
    layers.Dense(16, activation="relu"),
    layers.Dense(8, activation="relu"),
    layers.Dense(1),
])
model.summary()

# %% 4. REGRESSION WITH SYNTHETIC DATA
rng = np.random.default_rng(42)
X = rng.normal(size=(1000, 4)).astype("float32")
y = (2 * X[:, 0] - 1.5 * X[:, 1] + 0.5 * X[:, 2] + 0.2).astype("float32")

X_train, X_test = X[:800], X[800:]
y_train, y_test = y[:800], y[800:]

reg_model = keras.Sequential([
    layers.Input(shape=(4,)),
    layers.Dense(32, activation="relu"),
    layers.Dense(16, activation="relu"),
    layers.Dense(1),
])
reg_model.compile(optimizer="adam", loss="mse", metrics=["mae"])
history = reg_model.fit(
    X_train, y_train, validation_split=0.2,
    epochs=20, batch_size=32, verbose=0
)
print("Regression test metrics:", reg_model.evaluate(X_test, y_test, verbose=0))

plt.plot(history.history["loss"], label="train loss")
plt.plot(history.history["val_loss"], label="validation loss")
plt.title("Keras Regression Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.legend()
plt.show()

# %% 5. BINARY CLASSIFICATION
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X, y = make_moons(n_samples=1500, noise=0.20, random_state=42)
X = StandardScaler().fit_transform(X).astype("float32")
y = y.astype("float32")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

binary_model = keras.Sequential([
    layers.Input(shape=(2,)),
    layers.Dense(32, activation="relu"),
    layers.Dropout(0.15),
    layers.Dense(1, activation="sigmoid"),
])
binary_model.compile(
    optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
)
binary_model.fit(X_train, y_train, validation_split=0.2, epochs=15, verbose=0)
print("Binary test metrics:", binary_model.evaluate(X_test, y_test, verbose=0))
probs = binary_model.predict(X_test[:8], verbose=0).ravel()
print("Predicted probabilities:", np.round(probs, 3))

# %% 6. MULTICLASS CLASSIFICATION: MNIST
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

multiclass_model = keras.Sequential([
    layers.Input(shape=(28, 28)),
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.2),
    layers.Dense(10, activation="softmax"),
])
multiclass_model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
multiclass_model.fit(x_train, y_train, epochs=2, batch_size=128, validation_split=0.1)
print("MNIST test metrics:", multiclass_model.evaluate(x_test, y_test, verbose=0))

# %% 7. CALLBACKS
early_stopping = keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=3, restore_best_weights=True
)
reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6
)
checkpoint = keras.callbacks.ModelCheckpoint(
    "best_keras_model.keras", monitor="val_loss", save_best_only=True
)
# Use them in fit():
# model.fit(..., validation_split=0.2,
#           callbacks=[early_stopping, reduce_lr, checkpoint])

# %% 8. FUNCTIONAL API: MULTIPLE INPUTS
"""
The Functional API is useful when a model has multiple inputs, branches,
shared layers, or multiple outputs.
"""
numeric_input = keras.Input(shape=(3,), name="numeric_features")
category_input = keras.Input(shape=(2,), name="category_features")
combined = layers.Concatenate()([numeric_input, category_input])
hidden = layers.Dense(16, activation="relu")(combined)
output = layers.Dense(1, activation="sigmoid")(hidden)
functional_model = keras.Model(
    inputs=[numeric_input, category_input], outputs=output, name="multi_input_model"
)
functional_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
functional_model.summary()

# Tiny synthetic fit to demonstrate input structure.
numeric_demo = np.random.rand(64, 3).astype("float32")
category_demo = np.random.rand(64, 2).astype("float32")
labels_demo = np.random.randint(0, 2, size=(64, 1)).astype("float32")
functional_model.fit(
    [numeric_demo, category_demo], labels_demo, epochs=1, batch_size=16, verbose=0
)

# %% 9. CONVOLUTIONAL NEURAL NETWORK
cnn = keras.Sequential([
    layers.Input(shape=(28, 28, 1)),
    layers.Rescaling(1.0 / 255),
    layers.Conv2D(32, kernel_size=3, activation="relu"),
    layers.MaxPooling2D(),
    layers.Conv2D(64, kernel_size=3, activation="relu"),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(64, activation="relu"),
    layers.Dense(10, activation="softmax"),
])
cnn.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
# One epoch for a quick demonstration.
cnn.fit(x_train[..., None] * 255.0, y_train, epochs=1, batch_size=128, verbose=0)
print("CNN test metrics:", cnn.evaluate(x_test[..., None] * 255.0, y_test, verbose=0))

# %% 10. SAVE, LOAD, AND PREDICT
multiclass_model.save("keras_mnist_model.keras")
loaded_model = keras.models.load_model("keras_mnist_model.keras")
print("Loaded model prediction:", np.argmax(loaded_model.predict(x_test[:3], verbose=0), axis=1))

# %% 11. CUSTOM METRICS AND OPTIMIZERS
"""
Common optimizers: Adam, SGD, RMSprop, Adagrad.
Common metrics: accuracy, precision, recall, AUC, MAE.
Choose loss according to the problem and label encoding.
"""
custom_model = keras.Sequential([
    layers.Input(shape=(2,)),
    layers.Dense(8, activation="relu"),
    layers.Dense(1, activation="sigmoid"),
])
custom_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=[
        keras.metrics.BinaryAccuracy(name="accuracy"),
        keras.metrics.Precision(name="precision"),
        keras.metrics.Recall(name="recall"),
        keras.metrics.AUC(name="auc"),
    ],
)
print("Configured a model with additional evaluation metrics.")

# %% 12. REGULARIZATION AND OVERFITTING
"""
Useful tools:
- Dropout: randomly disables activations during training.
- L1/L2 regularization: penalizes large weights.
- EarlyStopping: stops when validation performance stops improving.
- Data augmentation: creates varied training images.
- Smaller models and more representative data can also reduce overfitting.
"""
regularized = keras.Sequential([
    layers.Input(shape=(4,)),
    layers.Dense(32, activation="relu", kernel_regularizer=keras.regularizers.l2(1e-4)),
    layers.Dropout(0.25),
    layers.Dense(1),
])
regularized.compile(optimizer="adam", loss="mse")

# %% 13. BEST PRACTICES
"""
- Normalize/standardize inputs appropriately.
- Avoid leakage from validation/test sets.
- Select the output activation and loss for the target format.
- Track validation metrics, not just training metrics.
- Save the complete model in .keras format.
- Use small experiments first; deep-learning training can be resource intensive.
- Never load model files from sources you do not trust.
"""

# %% 14. PRACTICE EXERCISES
"""
1. Change hidden-layer widths in the regression model.
2. Compare Adam and SGD on the same data.
3. Add EarlyStopping and compare training duration.
4. Try a CNN with different filter counts.
5. Build a Functional API model with two inputs.
6. Report precision, recall, and AUC for a binary task.
7. Save a model and write a separate script to load and predict.
"""
print("Keras guide finished.")

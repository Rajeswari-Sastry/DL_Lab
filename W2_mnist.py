# Commit separation: W2
# W2 - MNIST Classification using MLP

from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras import Input
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.utils import to_categorical
import os

os.makedirs("results", exist_ok=True)

# Load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()
print("Training data shape:", X_train.shape)
print("Example label (raw):", y_train[0])

# One-hot labels
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Model
model = Sequential([
    Input(shape=(28, 28)),   # ✅ modern way
    Flatten(),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train (fast demo)
history = model.fit(X_train, y_train, epochs=5, batch_size=64,
                    validation_data=(X_test, y_test), verbose=1)

# Evaluate
loss, acc = model.evaluate(X_test, y_test, verbose=1)
print(f"Test Accuracy: {acc:.4f} | Test Loss: {loss:.4f}")

# Plots
plt.plot(history.history['accuracy'], label='train acc')
plt.plot(history.history['val_accuracy'], label='val acc')
plt.legend(); plt.title('MNIST Accuracy'); plt.xlabel('epoch'); plt.ylabel('acc')
plt.savefig("results/mnist_acc.png", dpi=150, bbox_inches='tight')
plt.show()

plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.legend(); plt.title('MNIST Loss'); plt.xlabel('epoch'); plt.ylabel('loss')
plt.savefig("results/mnist_loss.png", dpi=150, bbox_inches='tight')
plt.show()
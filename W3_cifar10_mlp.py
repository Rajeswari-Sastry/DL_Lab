# Commit separation: W3
# W3 - Multi-Layer Perceptron on CIFAR-10

from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.utils import to_categorical
from keras.datasets import cifar10
from keras import Input
import matplotlib.pyplot as plt
import os, time

os.makedirs("results", exist_ok=True)

# Load data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Model
model = Sequential([
    Input(shape=(32, 32, 3)),   # ✅ modern Input layer
    Flatten(),
    Dense(1024, activation='relu'),
    Dense(512, activation='relu'),
    Dense(256, activation='relu'),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train
start = time.time()
history = model.fit(x_train, y_train, epochs=5, batch_size=64,
                    validation_data=(x_test, y_test), verbose=1)
end = time.time()
print(f"Training time (5 epochs): {(end-start):.2f} seconds")

# Evaluate
loss, accuracy = model.evaluate(x_test, y_test, verbose=1)
print(f"Test Accuracy: {accuracy:.4f} | Test Loss: {loss:.4f}")

# Plots
plt.plot(history.history['accuracy'], label="train acc")
plt.plot(history.history['val_accuracy'], label="val acc")
plt.legend(); plt.title("CIFAR10 MLP - Accuracy"); plt.xlabel("epoch"); plt.ylabel("acc")
plt.savefig("results/cifar10_mlp_acc.png", dpi=150, bbox_inches='tight')
plt.show()

plt.plot(history.history['loss'], label="train loss")
plt.plot(history.history['val_loss'], label="val loss")
plt.legend(); plt.title("CIFAR10 MLP - Loss"); plt.xlabel("epoch"); plt.ylabel("loss")
plt.savefig("results/cifar10_mlp_loss.png", dpi=150, bbox_inches='tight')
plt.show()
<<<<<<< HEAD
# W4 - CIFAR-10 using CNN (with 3 experiment presets)
# Change EPOCHS to 5 (demo), 32, 100, or 200 to feel slowness.
=======
# W4 - CIFAR-10 using CNN
>>>>>>> ce7761468e3d332753a4ccfb99afa2590a685769

from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from keras.datasets import cifar10
from keras.utils import to_categorical
<<<<<<< HEAD
=======
from keras import Input
>>>>>>> ce7761468e3d332753a4ccfb99afa2590a685769
import matplotlib.pyplot as plt
import os, time

os.makedirs("results", exist_ok=True)

<<<<<<< HEAD
=======
# Load data
>>>>>>> ce7761468e3d332753a4ccfb99afa2590a685769
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

<<<<<<< HEAD
# --- Experiment 3 (deeper CNN) as default: better accuracy, slower
EPOCHS = 5  # set 32/100/200 for your “slow laptop” proof
BATCH = 64

model = Sequential()
model.add(Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)))
model.add(MaxPooling2D((2,2)))
model.add(Conv2D(16, (3,3), activation='relu'))
model.add(MaxPooling2D((2,2)))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

start = time.time()
history = model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH, validation_data=(X_test, y_test), verbose=1)
end = time.time()
print(f"Training time ({EPOCHS} epochs): {(end-start):.2f} seconds")

loss, accuracy = model.evaluate(X_test, y_test, verbose=1)
print(f"Test Accuracy: {accuracy:.4f} | Test Loss: {loss:.4f}")

plt.plot(history.history['accuracy'], label='train acc')
plt.plot(history.history['val_accuracy'], label='val acc')
plt.legend(); plt.title("CIFAR10 CNN - Accuracy"); plt.xlabel("epoch"); plt.ylabel("acc")
plt.savefig("results/cifar10_cnn_acc.png", dpi=150, bbox_inches='tight')
plt.show()

plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.legend(); plt.title("CIFAR10 CNN - Loss"); plt.xlabel("epoch"); plt.ylabel("loss")
plt.savefig("results/cifar10_cnn_loss.png", dpi=150, bbox_inches='tight')
plt.show()
=======
# Hyperparameters
EPOCHS = 5
BATCH = 64

# Model
model = Sequential([
    Input(shape=(32,32,3)),
    Conv2D(32, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Conv2D(16, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train
start = time.time()
history = model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH,
                    validation_data=(X_test, y_test), verbose=1)
end = time.time()
print(f"Training time ({EPOCHS} epochs): {(end-start):.2f} seconds")

# Evaluate
loss, accuracy = model.evaluate(X_test, y_test, verbose=1)
print(f"Test Accuracy: {accuracy:.4f} | Test Loss: {loss:.4f}")

# Accuracy Plot
plt.plot(history.history['accuracy'], label='train acc')
plt.plot(history.history['val_accuracy'], label='val acc')
plt.legend(); plt.title("CIFAR-10 CNN - Accuracy"); plt.xlabel("epoch"); plt.ylabel("accuracy")
plt.savefig("results/cifar10_cnn_acc.png", dpi=150, bbox_inches='tight')
plt.show()

# Loss Plot
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.legend(); plt.title("CIFAR-10 CNN - Loss"); plt.xlabel("epoch"); plt.ylabel("loss")
plt.savefig("results/cifar10_cnn_loss.png", dpi=150, bbox_inches='tight')
plt.show()
>>>>>>> ce7761468e3d332753a4ccfb99afa2590a685769

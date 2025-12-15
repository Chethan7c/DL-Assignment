# -*- coding: utf-8 -*-
"""
Cats vs Dogs Classification
Modified for portability, robustness, and TensorFlow compatibility
"""

import os
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import random
import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# -------------------------------------------------
# Dataset configuration (ADDED)
# -------------------------------------------------
DATASET_DIR = "./cats_and_dogs_filtered"

if not os.path.exists(DATASET_DIR):
    raise FileNotFoundError(
        "Dataset not found. Please extract 'cats_and_dogs_filtered' into the project directory."
    )

# -------------------------------------------------
# Directory structure
# -------------------------------------------------
train_dir = os.path.join(DATASET_DIR, 'train')
validation_dir = os.path.join(DATASET_DIR, 'validation')

train_cats_dir = os.path.join(train_dir, 'cats')
train_dogs_dir = os.path.join(train_dir, 'dogs')
validation_cats_dir = os.path.join(validation_dir, 'cats')
validation_dogs_dir = os.path.join(validation_dir, 'dogs')

train_cat_fnames = os.listdir(train_cats_dir)
train_dog_fnames = os.listdir(train_dogs_dir)

print('Total training cat images:', len(train_cat_fnames))
print('Total training dog images:', len(train_dog_fnames))
print('Total validation cat images:', len(os.listdir(validation_cats_dir)))
print('Total validation dog images:', len(os.listdir(validation_dogs_dir)))

# -------------------------------------------------
# Display sample images (UNCHANGED)
# -------------------------------------------------
nrows, ncols = 4, 4
pic_index = 8

fig = plt.gcf()
fig.set_size_inches(ncols * 4, nrows * 4)

next_cat_pix = [os.path.join(train_cats_dir, fname)
                for fname in train_cat_fnames[pic_index-8:pic_index]]
next_dog_pix = [os.path.join(train_dogs_dir, fname)
                for fname in train_dog_fnames[pic_index-8:pic_index]]

for i, img_path in enumerate(next_cat_pix + next_dog_pix):
    sp = plt.subplot(nrows, ncols, i + 1)
    sp.axis('Off')
    img = mpimg.imread(img_path)
    plt.imshow(img)

plt.show()

# -------------------------------------------------
# Model definition (UNCHANGED)
# -------------------------------------------------
model = Sequential([
    Conv2D(16, (3,3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D(2,2),
    Conv2D(32, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.summary()

# -------------------------------------------------
# Model compilation (FIXED – deprecated lr removed)
# -------------------------------------------------
model.compile(
    optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# -------------------------------------------------
# Data generators
# -------------------------------------------------
train_datagen = ImageDataGenerator(rescale=1.0 / 255.0)
test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    batch_size=20,
    class_mode='binary',
    target_size=(150, 150)
)

validation_generator = test_datagen.flow_from_directory(
    validation_dir,
    batch_size=20,
    class_mode='binary',
    target_size=(150, 150)
)

# -------------------------------------------------
# Training
# -------------------------------------------------
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    steps_per_epoch=100,
    epochs=15,
    validation_steps=50,
    verbose=2
)

# -------------------------------------------------
# Prediction on a single image (OPTIONAL)
# -------------------------------------------------
fn = "cat2.jpg"  # change image name as needed
img_path = os.path.join(DATASET_DIR, fn)

if os.path.exists(img_path):
    img = load_img(img_path, target_size=(150, 150))
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x /= 255.0

    prediction = model.predict(x)[0][0]
    print(fn, "is a dog" if prediction > 0.5 else "is a cat")

# -------------------------------------------------
# Plot accuracy and loss
# -------------------------------------------------
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, label='Training Accuracy')
plt.plot(epochs, val_acc, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.figure()

plt.plot(epochs, loss, label='Training Loss')
plt.plot(epochs, val_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

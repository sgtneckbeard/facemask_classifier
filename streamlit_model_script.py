import numpy as np
import tensorflow as tf
import os
import cv2
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Activation
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# directory definition for training, validation, and test data
train_dir = 'data/train'
val_dir = 'data/validate'
test_dir = 'data/test'

# ImageDataGenerators for training, validation, and test data to load images and avoid maxing out memory
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# batch size for ImageDataGenerators
batch_size = 32

# create generators for training, validation, and test data
# generators will load images from the specified directories
# images will be resized to 224x300 and converted to grayscale
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 300),
    batch_size=batch_size,
    class_mode='categorical',
    color_mode='grayscale'
)
val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(224, 300),
    batch_size=batch_size,
    class_mode='categorical',
    color_mode='grayscale'
)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 300),
    batch_size=batch_size,
    class_mode='categorical',
    color_mode='grayscale'
)

# model architecture
IMG_SHAPE = 224
number_of_classes = train_generator.num_classes

model = Sequential()  # stack all the layers
# Convolutional layer block 1
model.add(Conv2D(32, kernel_size=(3, 3), input_shape=(224, 300, 1), padding='same'))
model.add(BatchNormalization())  # Batch Normalization layer
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Convolutional layer block 2
model.add(Conv2D(64, kernel_size=(3, 3), padding='same'))
model.add(BatchNormalization())  # Batch Normalization layer
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Convolutional layer block 3
model.add(Conv2D(128, kernel_size=(3, 3), padding='same'))
model.add(BatchNormalization())  # Batch Normalization layer
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Flattening the layers to feed into a DNN
model.add(Flatten())

# Dense layer with Batch Normalization
model.add(Dense(1024))
model.add(BatchNormalization())  # Batch Normalization layer
model.add(Activation('relu'))

# Dropout layer to reduce overfitting
model.add(Dropout(0.3))

# Output layer
model.add(Dense(number_of_classes, activation='softmax'))

# compile the neural network
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# model checkpoint definition to save the best model based on validation loss
model_checkpoint = ModelCheckpoint('best_model.keras',
                                    save_best_only=True,
                                    monitor='val_loss',
                                    verbose=1)

# early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=10)

# training the neural network
history = model.fit(train_generator,
                    validation_data=val_generator,
                    epochs=30,
                    shuffle=True,
                    batch_size=batch_size,
                    callbacks=[early_stopping, model_checkpoint])

# model evaluation on test data
test_loss, test_accuracy = model.evaluate(test_generator)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")

# training and validation loss plot
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('# epochs')
plt.ylabel('loss')
plt.legend()
plt.show()

# training and validation accuracy plot
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('# epochs')
plt.ylabel('accuracy')
plt.legend()
plt.show()

# Save the model
model.save('streamlit_model.h5')

model.summary()
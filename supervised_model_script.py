import numpy as np
import tensorflow as tf
import os
import cv2
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
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

# model architecture (based on Jadavpur article)
model = Sequential()
model.add(Conv2D(200, (3,3), input_shape=(224, 300, 1), activation='relu', name='conv2d_layer1'))
model.add(MaxPooling2D(pool_size=(3,3), name='maxpool_layer1'))
model.add(Conv2D(100, (3,3), activation='relu', name='conv2d_layer2'))
model.add(MaxPooling2D(pool_size=(3,3), name='maxpool_layer2'))
model.add(Flatten(name='flatten_layer'))
model.add(Dense(units=64, activation='relu', name='dense_layer1'))  # number of units in this layer is fixed to 128
model.add(Dropout(0.5, name='dropout_layer')) # dropout layer (Jadavpur article suggest it reduces overfitting)
model.add(Dense(4, activation='softmax', name='output_layer'))  
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# model checkpoint definition to save the best model based on validation loss
model_checkpoint = ModelCheckpoint(
    filepath = '20ep_basic_cnn_facemask_model_checkpoint.h5',
    monitor = 'val_loss',
    verbose = 1,
    save_best_only = True,
    mode = 'min' # for val_loss 
)

# training model
model_history = model.fit(
    train_generator,
    epochs = 20, 
    validation_data = val_generator,
    verbose = 1,
    callbacks=[model_checkpoint]
)

# model evaluation on test data
test_loss, test_accuracy = model.evaluate(test_generator)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")

# training and validation loss pl   ot
plt.plot(model_history.history['loss'], label='Training Loss')
plt.plot(model_history.history['val_loss'], label='Validation Loss')
plt.xlabel('# epochs')
plt.ylabel('loss')
plt.legend()
plt.show()

# training and validation accuracy plot
plt.plot(model_history.history['accuracy'], label='Training Accuracy')
plt.plot(model_history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('# epochs')
plt.ylabel('accuracy')
plt.legend()
plt.show()

# Save the model
model.save('50ep_xtraContrast_cnn_facemask_model.h5')

model.summary()
import os
from time import time
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.python.keras.callbacks import TensorBoard

train_dir = r'C:\Users\ragha\Desktop\New folder\gym.AI\Deep Learning\Food_Classification\data\train'
valid_dir = r'C:\Users\ragha\Desktop\New folder\gym.AI\Deep Learning\Food_Classification\data\val'

num_classes = len(os.listdir(train_dir))
INPUT_SHAPE = [32, 32, 3]

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

valid_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(INPUT_SHAPE[0], INPUT_SHAPE[1]),
    batch_size=64,
    color_mode='rgb',
    shuffle=True,
    class_mode='categorical')

valid_generator = valid_datagen.flow_from_directory(
    valid_dir,
    target_size=(INPUT_SHAPE[0], INPUT_SHAPE[1]),
    batch_size=64,
    shuffle=True,
    class_mode='categorical',
    color_mode='rgb')

res_base = ResNet50(weights='imagenet',
                    include_top=False,
                    input_shape=INPUT_SHAPE)
model = models.Sequential()
model.add(res_base)
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(num_classes, activation='softmax'))

model.summary()

early_stopping = EarlyStopping(monitor='val_loss', patience=30)
tensorboard = TensorBoard(log_dir="logs\{}".format(time()))
callbacks_list = [early_stopping, tensorboard]

model.compile(loss=tf.keras.losses.categorical_crossentropy,
              optimizer=tf.keras.optimizers.Adadelta(),
              metrics=['accuracy'])

STEP_SIZE_TRAIN = train_generator.n // train_generator.batch_size
STEP_SIZE_VALID = valid_generator.n // valid_generator.batch_size

history = model.fit_generator(train_generator,
                              steps_per_epoch=STEP_SIZE_TRAIN,
                              epochs=1,
                              shuffle=True,
                              callbacks=callbacks_list,
                              validation_data=valid_generator,
                              validation_steps=STEP_SIZE_VALID)

model.save(r'save_model\RESNET-50-FOOD-CLASSIFICATION.h5')
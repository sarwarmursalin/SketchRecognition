# -*- coding: utf-8 -*-
"""
Created on Sat Jan 18 23:58:36 2020

@author: Rukon
"""
#import subprocess

#subprocess.call(['pip', 'install', 'pydot'])
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import TensorBoard
import time
t = time.time()

Name = f"sketch_detection_cnn-{int(t)}"
#tensorboard = keras.callbacks.tensorboard_v1.TensorBoard(log_dir = f"C:\\Users\\Rukon\\Desktop\\sd_project\\logs\\{Name}", histogram_freq=1)

#converting images into grayscale



#initializing CNN
classifier = Sequential()

#convolution
classifier.add(Conv2D(32, (3, 3), activation = 'relu', input_shape=(128,128,3) ))
#pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

#adding 2nd colvolution layer
classifier.add(Conv2D(64, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

#flattening
classifier.add(Flatten())

#full connection
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 20, activation = 'softmax'))

#compiling CNN
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])


#part 2: Fittting CNN to images
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

train_set = train_datagen.flow_from_directory('dataset_sd/training_set',
                                              color_mode = 'rgb',
                                              target_size = (128,128),
                                              batch_size = 16,
                                              class_mode = 'categorical')

test_set = test_datagen.flow_from_directory('dataset_sd/test_set',
                                            color_mode = 'rgb',
                                            target_size = (128,128),
                                            batch_size = 16,
                                            class_mode = 'categorical')

history = classifier.fit_generator(train_set,
                         steps_per_epoch = 32,
                         epochs = 2,
                         validation_data = test_set,
                         validation_steps = 100)

#classifier.save('sketch_20_100.h5')
#plt.plot(history)
#plt.show()

# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()



dt = time.time() - t
print(f"Time spent in training: {dt/60:.2f} min.")











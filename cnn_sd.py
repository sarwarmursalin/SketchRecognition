# -*- coding: utf-8 -*-
"""
Created on Sat Dec 28 15:01:23 2019

@author: Rukon
"""
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 27 23:22:31 2019

@author: Rukon
"""
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
tensorboard = TensorBoard(log_dir = f"C:\\Users\\Rukon\\Desktop\\sd_project\\logs\\{Name}", histogram_freq=1)


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
classifier.add(Dense(units = 10, activation = 'softmax'))

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
                                              target_size = (128,128),
                                              batch_size = 16,
                                              class_mode = 'categorical')

test_set = test_datagen.flow_from_directory('dataset_sd/test_set',
                                            target_size = (128,128),
                                            batch_size = 16,
                                            class_mode = 'categorical')

history = classifier.fit_generator(train_set,
                         steps_per_epoch = 32,
                         epochs = 100,
                         validation_data = test_set,
                         validation_steps = 100)
classifier.save('sketch_10_2.h5')
plt.plot(history)
plt.show()



dt = time.time() - t
print(f"Time spent in training: {dt/60:.2f} min.")











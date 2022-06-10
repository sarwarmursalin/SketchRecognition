# -*- coding: utf-8 -*-
"""
Created on Sun Dec 29 00:11:51 2019

@author: Rukon
"""

#=======================using prediction==================
from keras.models import load_model
from keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2

"""
def load_image(img_path, show=False):

    img = cv2.imread(img_path)
    img = cv2.resize(img,(128,128))
    #imshow expects values in the range [0, 1]

    if show:
        plt.imshow(img,cmap='gray')                           
        plt.axis('off')
        plt.show()
    img=img.reshape(-1,128,128,3)
    return img

# load model
model = load_model("sketch_20_100.h5")

# image path
img_path = 'ant1.png'    # plane

# load a single image
new_image = load_image(img_path, True)

# check prediction
pred = model.predict(new_image)
label = ["AIRPLANE", "CLOCK", "ANT", "BANANA", "BELL", "BICYCLE", "CABIN",
         "CAR", "DOOR", "EYEGLASSES", "HOT AIR BALLOON", "HOURGLASS",
         "JACK-O-LANTERN", "KNIFE", "LIZARD", "PISTOL", "ROCKET",
         "SCISSORS", "STARFISH", "WINDMILL" ]
indx = np.argmax(pred)
print(label[indx])
print(pred)
for num in pred:
    i = 0
    for accu in num:
        print(f' {label[i]} -> {accu*100:.2f} %')
        i += 1
"""

#prediction 2
def load_image(img_path, show=True):

    img = image.load_img(img_path, target_size=(128, 128))
    img_tensor = image.img_to_array(img)              # (height, width, channels)
    img_tensor = np.expand_dims(img_tensor, axis=0)   # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
    img_tensor /= 255.                                # imshow expects values in the range [0, 1]

    if show:
        plt.imshow(img_tensor[0])                           
        plt.axis('off')
        plt.show()

    return img_tensor


if __name__ == "__main__":

    # load model
    model = load_model("sketch_20_100.h5")

    # image path
    img_path = 'test/cabin.png'    # plane
    #img_path = '/media/data/dogscats/test1/19.jpg'      # cat

    # load a single image
    new_image = load_image(img_path)

    # check prediction
    pred = model.predict(new_image)
    label = ["AIRPLANE", "CLOCK", "ANT", "BANANA", "BELL", "BICYCLE", "CABIN",
         "CAR", "DOOR", "EYEGLASSES", "HOT AIR BALLOON", "HOURGLASS",
         "JACK-O-LANTERN", "KNIFE", "LIZARD", "PISTOL", "ROCKET",
         "SCISSORS", "STARFISH", "WINDMILL" ]
    indx = np.argmax(pred)
    pred_dict = {}
    print(label[indx])
    pred_accu = []
    for num in pred:
        i = 0
        for accu in num:
            print(f' {label[i]} -> {accu*100:.2f} %')
            pred_accu.append(round(accu*100, 2))
            pred_dict[label[i]] = round(accu*100, 2)
            i += 1
            
#print(pred_dict)
pred_dict = sorted(pred_dict.items(), key=lambda x: x[1], reverse = True)
d = {}
cnt = 0
for key, value in pred_dict:
    if cnt<6:
        d[key] = value   
    cnt+=1

print(d)
        
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.bar(d.keys(), d.values())
#ax.bar(label,pred_accu)
plt.show()


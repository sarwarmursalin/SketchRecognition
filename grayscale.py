# -*- coding: utf-8 -*-
"""
Created on Sun Jan 26 16:57:21 2020

@author: Rukon
"""

import cv2
import glob, os, errno

mydir = r'C:\Users\Rukon\Desktop\sd_project\256x256\photo\tx_000000000000\cow'

#check if directory exist, if not create it
try:
    os.makedirs(mydir)
except OSError as e:
    if e.errno == errno.EEXIST:
        raise
for fil in glob.glob("*.jpg"):
    image = cv2.imread(fil) 
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # convert to greyscale
    cv2.imwrite(os.path.join(mydir,fil),gray_image) # write to location with same name
    
    

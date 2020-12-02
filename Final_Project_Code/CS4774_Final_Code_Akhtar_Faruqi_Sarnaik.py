'''
Class: CS 4774: Machine Learning Final Project
Title: Melanoma Classification
Team Member 1: Ramiz Akhtar (rsa5wj)
Team Member 2: Rayaan Faruqi (raf9dz)
Team Member 3: Kunaal Sarnaik (kss7yy)
Professor: Yanjun Qi
Due Date: December 6th, 2020
'''
#Guide: https://medium.com/ai-in-plain-english/image-processing-and-classification-with-python-and-keras-c368769bde26

#https://towardsdatascience.com/the-4-convolutional-neural-network-models-that-can-classify-your-fashion-images-9fe7f3e5399d

#Standard python imports
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import random
import imutils

# SKLearn 
from sklearn.utils import shuffle
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

# Keras Models
import keras
from keras.models import Sequential

# Keras Image Pre-Processing
    # https://machinelearningmastery.com/how-to-load-convert-and-save-images-with-the-keras-api/
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import array_to_img

#from keras.models import EfficientNet
    #https://keras.io/api/applications/efficientnet/
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.wrappers.scikit_learn import KerasClassifier


#https://medium.com/@ageitgey/python-3-quick-tip-the-easy-way-to-deal-with-file-paths-on-windows-mac-and-linux-11a072b58d5f
from pathlib import Path

#https://www.pluralsight.com/guides/importing-image-data-into-numpy-arrays
from PIL import Image #for converting images to grayscale, resizing image, trimming image

#OpenCV Library 
import cv2

#from google.colab.patches import csv2_imshow
random.seed(42)


#Method to obtain the data from the given path (see above for global paths)
def get_data(path_name):
    
    #Kunaal Path: "../PROJECT_DATA/skin-lesions"
    #Rayaan Path: "/Volumes/Rayaan_Ext2TB/MachineLearningProject/skin-lesions"
    #Ramiz Path: ""    

    #Kunaal Test Path: "../PROJECT_DATA/skin-lesions-trunc/skin-lesions-trunc"
    #Rayaan Test Path: "/Volumes/Rayaan_Ext2TB/MachineLearningProject/skin-lesions-trunc"
    #Ramiz Test Path: ""

    PATH = Path("../PROJECT_DATA/skin-lesions-trunc/skin-lesions-trunc")
    
    #join the path and the specific directory (train/valid/test) we're looking at
    split_path = os.path.join(PATH, path_name) 
    
    #Make relevant paths for the nevus, melanoma, and sbrk sub-directories in the current path
    nevus_path = os.path.join(split_path, "nevus")
    melanoma_path = os.path.join(split_path, "melanoma")
    sbrk_path = os.path.join(split_path, "seborrheic_keratosis")

    #loadImages will return an array of image arrays for each path 
    nevus_arr = loadImages(nevus_path)
    melanoma_arr = loadImages(melanoma_path)
    sb_arr = loadImages(sbrk_path)

    #set up classification matrices
    nevus_y = [0]*len(nevus_arr)
    melanoma_y = [1]*len(melanoma_arr)
    sb_y = [2]*len(sb_arr)
    
    #combine
    x = np.concatenate((nevus_arr, melanoma_arr, sb_arr), axis=0)
    y = nevus_y + melanoma_y + sb_y
    y = np.asarray(y)

    #shuffle the coupled data
    xData, yData = shuffle(x, y)

    #Return statement
    return xData, yData

    
# Method to loadImages in the given path (PRE-PROCESSES images as well!)
def loadImages(path):
    retArr = []
    print("Path is: ", path)
    width = 256
    height = 256
    for i in os.listdir(path):
        #Load each image in path
        path_i = os.path.join(path, i)
        img = cv2.imread(path_i, cv2.IMREAD_COLOR) 

        #Resizing images to 256 x 256
        #https://www.tutorialkart.com/opencv/python/opencv-python-resize-image/
        img_resized = cv2.resize(img, (width, height))

        #normalize image pixels from 0 to 1
        #https://stackoverflow.com/posts/39037135/edit    
        norm_image = cv2.normalize(img_resized, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
             
        #data augmentation (rotation)
        #https://www.pyimagesearch.com/2017/01/02/rotate-images-correctly-with-opencv-and-python/
        #Rotating in 360 over a continuous range versus in increments of 90???
        angle = random.randint(1,4)*90
        # print(angle)
        rotate_img = imutils.rotate(norm_image, angle)

        # cv2.imshow("Rotatated and resized image", rotated_img) #replace dot with underscore for collab

        rotate_img = np.asarray(rotate_img)
        retArr.append(rotate_img)
    retArr = np.asarray(retArr)
    print(retArr[0])
    return retArr

'''
#Method to create the CNN Model
def create_cnn(x, y, x_val, y_val, args=None):
    #create base CNN model
    model = Sequential()
    #hidden layer
    #can try activation as 'relu', 'softmax', 'sigmoid', 'tanh' 
    model.add(Dense(100), input_shape=(dim1,dim2), activation='relu')
    
    #output layer
    model.add(Dense(10), activation='softmax')

    #look at model summary
    model

    #compiling sequential model
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

    #train model for 3 epochs
    model.fit(x, y, batch_size=6, epochs=3, validation_data=(x_val, y_val))

    return model

#Method to train the CNN Model
def train_cnn(...):

    return model, history

#Method to train and select the CNN Model given the parameters specified.
def train_and_select_model(...):
    args = {
        'batch_size': ...,
        'num_epochs': ..., 
    }

    best_valid_acc = 0
    best_hyper_set = 0

    for learning_rate in [0.0001, 0.0005, 0.001, 0.005, 0.01]:
        for opt in ['adam', 'sgd']:
            for activation_func in ['softmax', 'relu', 'sigmoid', 'tanh']:
                #model = create_cnn

    return best_model, best_history, best_valid_acc, best_hyper_set

#Method to evaluate the model based on the train
def model_evaluation(...):
    
    return ...

def plot_history(...):

    return ...
'''

#MAIN METHOD
if __name__ == '__main__':

    train_path = "train"
    valid_path = "valid"
    test_path = "test"
    
    x_train, y_train = get_data(train_path)
    # x_valid, y_valid = get_data(valid_path)
    # x_test, y_test = get_data(test_path)
    
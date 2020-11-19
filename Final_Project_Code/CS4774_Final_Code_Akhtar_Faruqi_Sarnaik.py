# Class: CS 4774: Machine Learning Final Project
# Title: Melanoma Classification
# Team Member 1: Ramiz Akhtar (rsa5wj)
# Team Member 2: Rayaan Faruqi (raf9dz)
# Team Member 3: Kunaal Sarnaik (kss7yy)
# Professor: Yanjun Qi
# Due Date: December 6th, 2020

#Kunaal Path: "/c/Users/Kunaal/Documents/Fall 2020/CS 4774/PROJECT_DATA/skin-lesions"
#Rayaan Path: "/Volumes/Rayaan_Ext2TB/MachineLearningProject"
#Ramiz Path: ""

path = "/c/Users/Kunaal/Documents/Fall 2020/CS 4774/PROJECT_DATA/skin-lesions"

import numpy as np
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

import os
import sys
import pandas as pd
import keras
from keras.models import Sequential
#from keras.models import EfficientNet
#https://keras.io/api/applications/efficientnet/
#A potential model to use
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.wrappers.scikit_learn import KerasClassifier

imHeight = ... #image height in pixels
imWidth = ... #image width in pixels

#Method to obtain the data from the given path (see above for global paths)
def get_data(...):

    return xData, yData

#Method to create the CNN Model
def create_cnn(...):

    return model

#Method to train the CNN Model
def train_cnn(...):

    return model, history

#Method to train and select the CNN Model given the parameters specified.
def train_and_select_model(...):

    return best_model, best_history

#MAIN METHOD
if __name__ == '__main__':
    

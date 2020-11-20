# Class: CS 4774: Machine Learning Final Project
# Title: Melanoma Classification
# Team Member 1: Ramiz Akhtar (rsa5wj)
# Team Member 2: Rayaan Faruqi (raf9dz)
# Team Member 3: Kunaal Sarnaik (kss7yy)
# Professor: Yanjun Qi
# Due Date: December 6th, 2020


#Standard python imports
import numpy as np
import matplotlib.pyplot as plt

# SKLearn 
from sklearn.utils import shuffle
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

import os
import sys
import pandas as pd


# Keras Models
import keras
from keras.models import Sequential
#from keras.models import EfficientNet
    #https://keras.io/api/applications/efficientnet/
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.wrappers.scikit_learn import KerasClassifier

# Keras Image Pre-Processing
    # https://machinelearningmastery.com/how-to-load-convert-and-save-images-with-the-keras-api/
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import array_to_img

#https://medium.com/@ageitgey/python-3-quick-tip-the-easy-way-to-deal-with-file-paths-on-windows-mac-and-linux-11a072b58d5f
from pathlib import Path

#Method to obtain the data from the given path (see above for global paths)
def get_data(path_name):
    # Normalize image pixels from zero to 1
    # Remove mean and divide it by variance (NORMALIZATION)
        # Functions in tensorflow that will help do it automatically
    # Convert jpeg images to tensor or numpy input
    # Data Loader for mini-batches: https://www.tensorflow.org/api_docs/python/tf/keras/utils/Sequence
        # Can use any model with it
    # Can use data augmentation to improve accuracy (Rotation, Translation, etc. in tensorflow)
    # Don't have to resize images
        # If using MLP, will vectorize images
        # For CNN, same thing (input channel 3 for RGB)
            # KNN/linear Regression in final layer
    
    #Kunaal Path: "../PROJECT_DATA/skin-lesions"
    #Rayaan Path: "/Volumes/Rayaan_Ext2TB/MachineLearningProject"
    #Ramiz Path: ""

    PATH = Path("../PROJECT_DATA/skin-lesions")
    #PATH = os.path.abspath("..\PROJECT_DATA\skin-lesions")
    
    #join the path and the specific directory (train/valid/test)we're looking at
    split_path = os.path.join(PATH, path_name) 
    
    #Make relevant paths for the nevus, melanoma, and sbrk sub-directories in the current path
    nevus_path = os.path.join(split_path, "nevus")
    melanoma_path = os.path.join(split_path, "melanoma")
    sbrk_path = os.path.join(split_path, "seborrheic_keratosis")

    #loadImages will return an array of image arrays for each path 
    nevus_arr = loadImages(nevus_path)
    # melanoma_arr = loadImages(melanoma_path)
    # sb_arr = loadImages(sbrk_path)
    
    '''
    INSERT PREPROCESSING/DATA AUGMENTATION
    '''
    
    #set up classification matrices
    nevus_y = [0]*len(nevus_arr)
    # melanoma_y = [1]*len(melanoma_arr)
    # sb_y = [2]*len(sb_arr)
    
    #Combine and Shuffle
    

    #Extract the xdata and ydata from the shuffled, combined data matrix


    #Return statement
    #return xData, yData
    return nevus_arr
    
# Method to loadImages in the given path (PRE-PROCESSES images as well!)
def loadImages(path):
    retArr = []
    print("Path is: ", path)
    for i in os.listdir(path):
        #Load each image in path
        print("File is: ", i)
        img = load_img(os.path.join(path, i))
        img_arr = img_to_array(img)

        #Pre-processing

        
        retArr.append(img_arr)
    retArr = np.asarray(retArr)
    return retArr

'''
#Method to create the CNN Model
def create_cnn(...):

    return model

#Method to train the CNN Model
def train_cnn(...):

    return model, history

#Method to train and select the CNN Model given the parameters specified.
def train_and_select_model(...):

    return best_model, best_history

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
    
    # x_train, y_train = get_data(train_path)
    # x_valid, y_valid = get_data(valid_path)
    # x_test, y_test = get_data(test_path)

    nevus = get_data(train_path)
    print(nevus)
    

    

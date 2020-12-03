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
#import imutils

# SKLearn 
from sklearn.utils import shuffle
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

# Keras Models
import keras
from keras.models import Sequential

# Keras Image Pre-Processing
    # https://machinelearningmastery.com/how-to-load-convert-and-save-images-with-the-keras-api/
    #https://machinelearningmastery.com/how-to-configure-image-data-augmentation-when-training-deep-learning-neural-networks/
from keras.preprocessing.image import load_img, img_to_array, array_to_img, ImageDataGenerator

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

from statistics import median, mean

#from google.colab.patches import csv2_imshow
random.seed(42)


#Global Variables
#Kunaal Path: "../PROJECT_DATA/skin-lesions"
#Rayaan Path: "/Volumes/Rayaan_Ext2TB/MachineLearningProject/skin-lesions"
#Ramiz Path: ""    

#Kunaal Test Path: "../PROJECT_DATA/skin-lesions-trunc/skin-lesions-trunc"
#Rayaan Test Path: "/Volumes/Rayaan_Ext2TB/MachineLearningProject/skin-lesions-trunc"
#Ramiz Test Path: ""

PATH = Path("../PROJECT_DATA/skin-lesions-trunc/skin-lesions-trunc")

dim = 128

IMG_WIDTH = dim
IMG_HEIGHT = dim
#maybe use the median size in the dataset???

#Function to get average image size (x pixels vs y pixels) for all images in the set (RUN ONLY ONCE)
def get_average_image_size():
    widths = []
    heights = []
    img_count = 0

    size_arr = []


    split_path_list = [os.path.join(PATH, "train"), os.path.join(PATH, "valid"), os.path.join(PATH, "test")]
    n_mel_sbrk_path_list = ["nevus", "melanoma", "seborrheic_keratosis"]
    
    for split_path in split_path_list:
        for n_mel_sbrk_path in n_mel_sbrk_path_list:
            path = os.path.join(split_path, n_mel_sbrk_path)
            for i in os.listdir(path):
                image_path = os.path.join(path, i)
                image = load_img(image_path, grayscale = True)
                
                img_size = image.size
                widths.append(img_size[0])
                heights.append(img_size[1])
                img_count += 1
                                
    #get median 
    med_width = int(median(widths) / 8)
    med_height = int(median(heights) / 8)
    
    return med_width, med_height, img_count

#Method to obtain the data from the given path (see above for global paths)
def get_data(path_name):
        
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
    for i in os.listdir(path):
        #Load each image in path
        path_i = os.path.join(path, i)
        img = cv2.imread(path_i, cv2.IMREAD_COLOR) 

        #Resizing images to 256 x 256
        #https://www.tutorialkart.com/opencv/python/opencv-python-resize-image/
        img_resized = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))

        #normalize image pixels from 0 to 1
        #https://stackoverflow.com/posts/39037135/edit    
        norm_image = cv2.normalize(img_resized, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        
        #data augmentation (rotation)
        #https://www.pyimagesearch.com/2017/01/02/rotate-images-correctly-with-opencv-and-python/
        #Rotating in 360 over a continuous range versus in increments of 90???
        # angle = random.randint(1,4)*90
        # print(angle)
        # rotate_img = imutils.rotate(norm_image, angle)


        #FOR KERAS
        '''
        try using Keras deep learning data augmentation libraries 
        https://keras.io/api/preprocessing/image/
        https://towardsdatascience.com/data-augmentation-in-medical-images-95c774e6eaae 

        Consider upsampling and downsampling by finding median image size in the dataset

        AWS free credits
        Use Rivanna with SLURM scripts and python files
        '''

        # cv2.imshow("Rotatated and resized image", rotated_img) #replace dot with underscore for collab

        rotate_img = np.asarray(norm_image)
        retArr.append(norm_image)
    retArr = np.asarray(retArr)
    print(retArr[0])
    return retArr


#Method to create the CNN Model
def create_and_train_cnn(x_train, y_train, x_val, y_val, args=None):
    #create base CNN model --- technically only an NN model until we add Conv2D Layers

    print("DATA SIZES BEFORE------------")
    print(np.shape(x_train))
    print(np.shape(y_train))

    print(np.shape(x_val))
    print(np.shape(y_val))


    x_train = x_train.reshape(len(x_train), dim, dim, 3)
    x_val = x_val.reshape(len(x_val), dim, dim, 3)

    y_train = keras.utils.to_categorical(y_train, num_classes=3)
    y_val = keras.utils.to_categorical(y_val, num_classes=3)

    # y_train = y_train.reshape(len(x_train), 1)
    # y_val = y_val.reshape(len(x_val), 1)
    
    input_shape = (dim, dim, 3)


    print("DATA SIZES AFTER------------")
    print(np.shape(x_train))
    print(np.shape(y_train))

    print(np.shape(x_val))
    print(np.shape(y_val))

    #If no arguments are passed in, set to default values
    if (args == None):
        args = {}
        args['batch_size'] = 3
        args['epochs'] = 15
        args['opt'] = 'adam'

    ###DEFINE MODEL ARCHITECTURE
    model = Sequential()

    #can try activation as 'relu', 'softmax', 'sigmoid', 'tanh' 
    model.add(Conv2D(filters=dim, activation='relu', kernel_size=3, input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(18,18), strides=(1,1)))
    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(units=3, activation='relu'))
    #model.add(Dense(100, input_shape=(IMG_WIDTH, IMG_HEIGHT, 3), activation='relu'))
    
    #output layer
    #model.add(Dense(10, activation='softmax'))

    #look at model summary
    model.summary()

    #compiling sequential model
    #can try optimizer as 'adam', 'sgd', 'adadelta', 'adagrad', 'nadam', 'adamax', and more
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=args['opt'])          #Review what is the best loss function for this

    #create generator and iterator
    generator = ImageDataGenerator()
    itr = generator.flow(x_train, y_train)

    #train model for given epochs with given batch_size
    history = model.fit(x=itr, batch_size=args['batch_size'], epochs=args['epochs'], validation_data=(x_val, y_val))

    return model, history
    
#Method to train and select the CNN Model given the parameters specified.
def train_and_select_model(x_train, y_train, x_validation, y_validation):
    args = {
        'batch_size': 8,
        'epochs': 8, 
    }

    best_valid_acc = 0
    best_hyper_set = {}


    for learning_rate in [0.0001, 0.0005, 0.001, 0.005, 0.01]:
        for opt in ['adam', 'sgd', 'adagrad', 'nadam', 'adamax']:
            for activation_func in ['softmax', 'relu', 'sigmoid', 'tanh']:
                args['learning_rate'] = learning_rate
                args['opt'] = opt
                args['activation_func'] = activation_func

                print("Creating and training model with learning rate ", learning_rate,
                 ", optimizer, " opt, ", activation function, ", activation_func)

                model, history = create_and_train_cnn(x_train, y_train, x_validation, y_validation, args)

                validation_accuracy = history.history['val_accuracy']

                max_validation_accuracy = max(validation_accuracy)
                if max_validation_accuracy > best_valid_acc:
                    best_model = model
                    best_history = history
                    best_valid_acc = max_validation_accuracy
                    best_hyper_set['learning_rate'] = learning_rate
                    best_hyper_set['opt']  = opt
                    best_hyper_set['activation_function'] = activation_func
        
    return best_model, best_history, best_valid_acc, best_hyper_set

#Method to evaluate the model based on the train
def model_evaluation(...):
    
    return ...

def plot_history(...):

    return ...


#MAIN METHOD
if __name__ == '__main__':
    # x_size, y_size, num_images = get_average_image_size()
    # print("------------------Median Image Size-----------------------")
    # print(x_size)
    # print(y_size)
    # print("Image Count: ", num_images)


    train_path = "train"
    valid_path = "valid"
    test_path = "test"
    
    x_train, y_train = get_data(train_path)
    x_valid, y_valid = get_data(valid_path)
    # x_test, y_test = get_data(test_path)
    
    #test_model, test_history = create_and_train_cnn(x_train, y_train, x_valid, y_valid)
    b_model, b_history, b_valid_acc, b_hyper_set = train_and_select_model(x_train, y_train, x_valid, y_valid)

    print("\nLet's Go.\n")
    print("Best model summary: ", b_model.summary())
    # print("Best history: ", b_history)
    print("Best validation accuracy: ", b_valid_acc)
    print("Best Hyper Set: ", b_hyper_set)
    
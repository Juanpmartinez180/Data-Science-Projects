### Library import

import sys
import os
import serial
import csv
import time
import pandas as pd
import functools
import numpy as np
import random as rd
import matplotlib
import threading
from numpy import diff
import seaborn as sns

import matplotlib.pyplot as plt

from matplotlib.pyplot import figure

plt.rcParams["figure.figsize"] = (20,3)

#---- CNN model libraries ----
import keras 
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from keras import callbacks
from keras import backend as K
from keras.models import load_model

import helpers
import object_location

### Functions

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


### Load ML model
def load_ml_model(path_to_load):
    #load model from path
    path_to_load = '../models/model_v2.h5'
    #define needed dependecies to load the model
    dependencies = {
        'f1_m': f1_m
    }
    #load the model
    classifier = load_model(path_to_load, custom_objects=dependencies)
    
    print("Machine Learning model succesfully loaded!")
    
    return classifier

def capture_data(samples_to_capture, classifier, print_signature, serial_port):
    
    ### Capture data
    # Define constants
    #----samples_to_capture = 1  # Number of samples to capture
    serial_length = 2048  # Length of the serial comm message
    n_sensors = 3  #  Number of sensors

    ### Initiate serial port communications
    ser = serial.Serial(serial_port, baudrate=115200)
    ser.flushInput()

    ### Clear data tensor
    received_data = np.zeros([samples_to_capture,  # Define tensor to store captured data
                              n_sensors,
                              serial_length])

    for i in range(samples_to_capture):
        for j in range(n_sensors):
            temp_data = []  # To store temporal data

            for k in range(serial_length):
                line = ser.readline()  # Read serial port input data
                if line:
                    string = line.decode(errors='ignore') # Decode serial data
                    temp_data.append(string.split('\r')[0])  # Save data to temporal storage

            temp_data = [2.5 if (len(data)!=4) else data for data in temp_data]  # Replace all the values with (len=!4) with 2.5 (avg value)
            received_data[i, j, :] = temp_data  # Move data from temporal storage to the tensor

    ser.close()  # Close serial communiacations
    
    
    curated_data = np.zeros([samples_to_capture, n_sensors, 81])
    plot_processing = False

    for i in range(samples_to_capture):
        for j in range(n_sensors):

            ## Define sample to analize

            sample = received_data[i-1, j-1, 100:].astype(float)

            ## Apply derivate and noise reduction techniques

            sample_denoised = helpers.derivate_and_noise_reduction(sample, plot_processing)

            ## Detect pulses and retrieve related sample indexes

            center_point_pulse, center_point = helpers.pulse_detection(sample_denoised, plot_processing)

            ## Transform sample dimention to match ML algorithm input dimention

            output_space = helpers.dimention_transformation(center_point_pulse, center_point, plot_processing)

            ## Replace calculated indexes in the final array
            for pulse_idx in output_space:
                curated_data[i-1, j-1, pulse_idx] = 1
                
            if print_signature:
                print('Output space indexes: ',output_space)

        if print_signature:
            sns.heatmap(curated_data[i-1, :, :])
            plt.xlabel('Index')
            plt.ylabel('Sensor')
            plt.title('Echo signature')
            plt.show()

    ### Reshape 3D data into 2D array
    model_input  = curated_data.reshape(curated_data.shape[0],
                          (curated_data.shape[1] * curated_data.shape[2]))
    
    if print_signature:
        print("Original input data shape: ", curated_data.shape)
        print("New input data shape: ", model_input.shape)

    ### Make predictions
    X_val = model_input  #Values to predict

    #define a threshold value 
    threshold = 0.3
    #ANN prediction
    y_pred = classifier.predict(X_val)
    #keep predictions above the threshold
    y_pred = (y_pred > threshold)

    for pred in range(len(y_pred)):

        print('\nPredicted object quadrant: ',
              np.argwhere(np.isin(y_pred[pred], True)).ravel())
        predicted_points = np.argwhere(np.isin(y_pred[pred], True)).ravel()

        plt.rcParams["figure.figsize"] = (6,6)
        object_location.test(predicted_points, predicted_points)

        
        
    return
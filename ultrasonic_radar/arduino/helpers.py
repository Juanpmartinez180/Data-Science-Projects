import pandas as pd
import functools
import numpy as np
from numpy import diff
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from numpy import diff
from scipy.signal import hilbert, find_peaks, savgol_filter


############

def echo_center_point(sample, threshold):
    """
    Description: Function to return the sample number for a detected initial pulse and for the captured echo/es
    Input: sample: Array with all the raw data
            threshold: Value that the mean-criteria needs to meet to consider the sample as a pulse
    Output: echo_center: Sample number/numbers related to the detected echo/echoes
            pulse_center: Sample number related to the detected initial pulse
    """
    
    pulse_window = 7 # Number of samples to compute
    
    echo_values = [] # Sample indexes that meet the threshold (for the received echoes)
    echo_center = [] # Sample indexes for echoes - preprocessed

    pulse_values = [] # Sample indexes that meet the threshold (for the initial pulse)
    pulse_mean_values = [] # Mean values for the computed sample window
    
    
    for i in range(len(sample)):       
        sample_window = sample[i:i+pulse_window]    # Samples within desired window
        window_mean = sum(sample_window)/len(sample_window)  # Window mean value
        if window_mean > threshold: 
            if i > 450:                # index 500 represents the minimum distance a echo pulse could be find
                echo_values.append(i)   # Save the sample index if the sample mean meets the threshold and the index is bigger than 500.
            else:
                pulse_values.append(i)  # For idx smaller than 500 a initial pulse should be existing
                pulse_mean_values.append(window_mean) # Save the sample mean values
                
                
    pulse_idx = pulse_mean_values.index(max(pulse_mean_values)) # Index for the maximum value reached by the pulse
    pulse_center = pulse_values[pulse_idx] # Sample index for the initial pulse center (triggered by the sensor)
    
    pulse_width = 80 #int(len(pulse_values)*0.4)  # Define the maximum width for the pulse
    
    for i in range(0, round(len(echo_values)/pulse_width)):
        temp_values = echo_values[pulse_width*(i):pulse_width*(i+1)]  # Echoes values within the pulse width
        temp_center = sum(temp_values)/len(temp_values)  # Compute the pulse center as the mean within the pulse width
        echo_center.append(int(temp_center))  # Save echoes center
        
    return(echo_center, pulse_center)

############

def echo_distance(p2, p1):
    """
    Description: Function to return the distance between 2 points, taking into account the air speed and sample frequency of the device
    Input: p1: First point
            p2: Second point
    Output: Distance between p1 and p2
    """
    air_speed = 343   # Air speed in m/s
    sample_freq = 140000   # Sample frequency of the device, in Hz
    
    threshold_values = p2-p1  # Amount of samples between the points
    distance = (1/sample_freq)*threshold_values*air_speed/2  # Convert the samples to distance in [m]
    
    return(distance)

############

def derivate_and_noise_reduction(sample, print_results):
    ## Derivate

    dx = 1 # Define derivate step
    dy = np.abs(diff(sample)/dx) # Derivate the raw data and take absolute value
    
    if print_results:
        plt.plot(np.arange(0, len(dy)), dy, label = 'Derivate function')  # Plot the derivate function

    ### Noise reduction

    limit = max(dy)  # Calculate max value
    threshold = 0.2  # Set threshold value
    dy = [0 if (val < threshold*limit) else val for val in dy]  # All values below (threshold*limit) will be turned into 0 

    ### Plot results
    if print_results:  
        plt.plot(np.arange(0, len(dy)), dy, label = "Derivate w/ noise reduction")
        plt.plot(np.arange(0, len(dy)), [threshold*limit for i in range(len(dy))], label = "Noise reduction threshold")

        plt.title('Before and after derivate and noise reduction techniques')
        plt.xlim([0,2048])
        plt.legend(loc = 'upper right')
        plt.show()
    
    return(dy)

############

def pulse_detection(sample, print_results):
    
    if print_results: 
        ax_y = np.arange(0, len(sample))
        plt.plot(ax_y, sample, label = "Echo signature")  #Plot sample data
    
    ### Define a confidence threshold to compute the mean-criteria to detect the pulses

    threshold = 0.12*max(sample)
    confidence_threshold = [threshold for i in range(len(sample))]
    
    if print_results:
        plt.plot(ax_y,confidence_threshold, color = 'red', label = 'Confidence threshold')

    ### Calculate the echo/echoes center
    
    center_point = echo_center_point(sample, threshold)[0]
    
    if print_results:
        echo_plot_y = [i for i in range(int(1.2*max(sample)))]

        for i in range(len(center_point)): 
            echo_plot_x = [center_point[i] for j in range(len(echo_plot_y))]
            plt.plot(echo_plot_y, echo_plot_x, label = 'Echo center point')

    ### Calculate the pulse center

    center_point_pulse = int(echo_center_point(sample, threshold)[1])
    
    if print_results:
        echo_plot_x = [center_point_pulse for i in range(len(echo_plot_y))]
        plt.plot(echo_plot_x, echo_plot_y, label = 'Pulse center point')

    ### Plot and print results
    if print_results:

        print('Pulse center point: ', center_point_pulse)
        print('Echo center point: ', center_point)

        plt.ylim([0, 1.2*max(sample)])
        plt.legend(loc = "lower right")
        plt.title('title')
        plt.show()

        i = 0
        for echo_point in center_point:
            print('Echo distance', i, '->' , echo_distance(echo_point, center_point_pulse), '[m]')
            i += 1
    
    return(center_point_pulse, center_point)

############

def dimention_transformation(center_point_pulse, center_point, print_results):
    # Define constants

    initial_sample_freq = 140000  # ADC space
    final_sample_freq = 6800  # ML algorithm space

    transformation_ratio = final_sample_freq / initial_sample_freq

    output_space = [int((echo - center_point_pulse)*transformation_ratio) for echo in center_point]

    return(output_space)

############

def output_dimention_pulses(sample, threshold):
    """
    Description: Function to return the sample number for a detected initial pulse and for the captured echo/es
    Input: sample: Array with all the raw data
            threshold: Threshold value used for the promincence criteria
    Output: output_space
    """
    initial_sample_freq = 140000  # ADC space
    final_sample_freq = 6800  # ML algorithm space
    default_detection_threshold = 0.1
    
    transformation_ratio = final_sample_freq / initial_sample_freq

    if threshold > 0.001:
        prominence = threshold
    else:
        prominence = default_detection_threshold

    signal = sample
    # Obtener la envolvente de la señal usando la Transformada de Hilbert
    analytic_signal = hilbert(signal)
    envelope = np.abs(analytic_signal)
    
    baseline = np.median(envelope)
    envelope_zero_centered = envelope - baseline

    # --- 3. Detección de Picos en la Envolvente ---
    
    # Encuentra picos que tengan una altura mínima de 0.1 por encima del nivel base y
    # que estén separados por al menos 150 muestras.
    peaks, properties = find_peaks(envelope_zero_centered, prominence=prominence, distance=150)

    if peaks[0] < 400:
        initial_pulse = peaks[0]
        peaks_clean = peaks[~(peaks < 500)]
    else:
        initial_pulse = 0
        
    output_space = [int((echo - initial_pulse)*transformation_ratio) for echo in peaks_clean]

    return(output_space, peaks)

        
    return(output_space)
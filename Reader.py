# -*- coding: utf-8 -*-
"""
@author: Artur Smiechowski
Changed the structure to not use nested methods, if I want to compartmentalize data that's what OOP is for
"""

#%% import packages
import numpy as np
import matplotlib.pyplot as plt
import serial
import datetime, pytz
import os

#%%
def initialize_plot():
    '''initialize_plot
    Initializes the plot for action display
    
    Parameters
    NONE
    Returns
    NONE
    '''
    # Creates the data range for action vs time
    display_data_y = np.array([0,1,0,2,0,3]*10)
    display_data_x = np.arange(0,60)
    # Creates the step plot
    plt.step(display_data_x,display_data_y,where='post') # Post fixes aligment
    
    actions=['rest','rock','paper','scissors']
    plt.xlim([-1,60]) # set the x axis limits
    plt.ylim([0,len(actions)-0.75]) # set the y axis limits 
    
    plt.xlabel('time') # add x axis label
    plt.yticks([0,1,2,3],labels=actions)
    plt.title('Time Plot of Actions') # add title 
    #plt.legend(('ch1','ch2','ch3')) # add legend with data from A0 as ch1, A2 as ch2, A3 as ch3
    
def initialize_arrays(recording_duration,n_channels,fs):
        '''
        this function initializes two arrays. sample_time is a 1d array that holds the 
        time of each sample recorded. sample_data is a 2D array with n_channels. Each 
        column is a seperate channels. the channels correspond to sensors on the Arduino

        Parameters
        ----------
        recording_duration : int
            Length of time to run the recording
        n_channels : int
            number of sensors hooked up to the Arduino
        fs : int
            the sampeling frequency of the arduino 

        Returns
        -------
        sample_time : 1D array
            holds the time of each sample
        sample_data: 2D array
            Holds the reading from each sample in each column

        '''
        sample_time = np.zeros([recording_duration*fs,1])*np.nan # create #sample by 1 array
        sample_data = np.zeros([recording_duration*fs,n_channels])*np.nan # create #sample by number of channels array
        
        return sample_time, sample_data # Was the empty return from before dumping all local variables?

def record_data(com_port='COM4', recording_duration=60, n_channels=3, fs=500, out_string='.'):
    '''record_data
    Records emg data from multiple channels for given time
    
    Parameters
    com_port : string ~ specifices the port the emg device is on
    recording_duration ~ int ~ the recording period in seconds
    n_channels ~ int ~ the number of channels/devices being recorded
    fs ~ int ~ the emg sampling frequency
    out_string ~ string ~ the file location to save to
    
    Returns
    NONE
    '''
    # Initialize the plot
    initialize_plot()
    plt.show() # show the plot
    print('rest, rock, paper, scissors') 
    print('Hold each action for 1s')
   
    # Initialize the arrays for reading data in
    sample_time, sample_data= initialize_arrays(recording_duration, n_channels, fs)
    
    prompt=input('Press enter to start:') # Just use input to wait for user, why check the input?
       
    with serial.Serial(port=com_port, baudrate=500000) as arduino_data: #open connection to serial monitor. takes input com_port form main function
        
            
            n_channel=len(sample_data[0]) #create variable that is the number of columns in sample_data
            for data_index in range(len(sample_data)): # for loop to index data into arrays
                try:
                    arduino_string=arduino_data.readline().decode('ascii') # read line from serial
                    arduino_list=arduino_string.split() # split string into list at space
                    sample_time[data_index]=int(arduino_list[0])/1000 # put 1st piece of data into time array
                       
                    for channel_index in range(n_channel): # for loop to index data into the corrct column of sample_data and data_line
                        sample_data[data_index,channel_index]=int(arduino_list[channel_index+1])*5/1024          
                except: # On dropped data ignore
                    pass
                               
    arduino_data.close() # close searial port
    '''
    #Save data 
    try:
            os.mkdir(out_string)
    except OSError:
            print('save failed')
    else:
            print('folder created')
    '''
    
    np.save('Smiechowski_RPS_Data',sample_data) #save sample_data
    np.save('Smiechowski_RPS_Time',sample_time) #save sample_time


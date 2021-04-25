"""
@author: Artur Smiechowski
"""
# %% Imports
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing as skl
# To import my own module requires this mess
import sys
#sys.path.insert(1, r'C:\Users\Artur Smiechowski\Documents\BME227_Code\Lab_5_BME_227')
sys.path.insert(1, r'F:\Python_Projects\Lab_5_BME_227') # Working across 2 machines have different paths for each
import Reader

# %% Part 1
''' --- Commented for now to allow for full code run without error
# Call the main reading method from the modified part 2 code
Reader.record_data(com_port='COM3', recording_duration=60, n_channels=3, fs=500, out_string='.')
'''

# %% Part 2

def load_data(name):
    '''load_data
    
    '''
    emg_data = np.load(str(name) + '_RPS_Data.npy') # Load in data
    emg_time = np.load(str(name) + '_RPS_Time.npy') # Load in time
    
    return emg_data, emg_time # Return loaded data and time

def epoch_data(emg_data, fs, epoch_duration):
    '''epoch_data
    
    '''
    # Subtract the mean offset from each channel
    for channel in range(np.size(emg_data, 1)): # Double check the array dimensions
        mean = np.mean(emg_data[:,channel]) # Find the mean of the current channel
        emg_data[:,channel] -= mean # Subtract the mean across the current channel
        #--- Fun fact, -= across an array only works for  numpy arrays, not on lists
    
    # epoch_count defined as variable for readability
    epoch_count = int((np.size(emg_data, 0) / (fs*epoch_duration)) + 0.5) # int and + 0.5 to account for uneven divsion
    # Initialize epoched_data array
    #epoched_data = np.zeros(epoch_count, fs*epoch_duration, np.size(emg_data, 1))
    epoched_data = np.reshape(emg_data, (epoch_count,fs*epoch_duration,np.size(emg_data, 1)))
    
    return epoched_data # Return the epoched_data array

def extract_features(epoched_data):
    '''extract_features

    '''
    # Initialie the feature sub-matrices
    epoch_var = np.zeros([np.size(epoched_data, 0), np.size(epoched_data, 2)]) # variance
    epoch_mav = np.zeros([np.size(epoched_data, 0), np.size(epoched_data, 2)]) # mean absolute value
    epoch_zc = np.zeros([np.size(epoched_data, 0), np.size(epoched_data, 2)]) # number of zero crossing events
    
    for epoch in range(np.size(epoched_data, 0)): # iterate through epochs
        for channel in range(np.size(epoched_data, 2)): # iterates through channels
            # Set the values of the variance and MAV sub-matrices
            epoch_var[epoch, channel] = np.nanvar(epoched_data[epoch, :, channel])
            epoch_mav[epoch, channel] = np.mean(np.abs(epoched_data[epoch, :, channel]))
            
            zc = 0 # zero crossing count
            prev_sign = np.sign(epoched_data[epoch,0,channel]) # Pulls the sign of the first sample
            for sample in range(np.size(epoched_data, 1)): # Iterates through all the samples in the current epoch and channel
                if np.sign(epoched_data[epoch,sample,channel]) != prev_sign: # Check if the sign of the current value is different than previous sign
                    zc += 1 # Increment zc when sign is different
                prev_sign = np.sign(epoched_data[epoch,sample,channel])
            epoch_zc[epoch, channel] = zc # set the epoch and channel zero crossing count to zc
        
    # Concatenate the submatrices column-wise into the full feature array
    features = np.concatenate((epoch_var,epoch_mav,epoch_zc),axis=1)
    
    # Normalize the feature array
    features = skl.scale(features)
    #--- Cannot get mean and std exactly 0 and 1. Tested on smaller arrays with same syntax and was succesful
    #--- Output mean and std where X*10^-17 and 0.99999... respectively, assuming floating point error or issue with big array
    
    # Create the shorthand list
    feature_shorthands = ['Var_ch0','Var_ch1','Var_ch2','MAV_ch0','MAV_ch1','MAV_ch2','ZC_ch0','ZC_ch1','ZC_ch2'] # Why is this even in this method?
    
    return features, feature_shorthands # Return the normalized feature array

def make_truth_data(action_sequence, epochs_per_action):
    '''make_truth_data
    
    '''
    instructed_action = [] # Create the empty list for instructed actions
    for action in range(len(action_sequence)): # I guess we're assuming non-fractional values at this point? (I'd code around that but time)
        for epoch in range(epochs_per_action): # This could probably be done more elegantly but...
            instructed_action.append(action_sequence[action]) # Append the current action epochs_per_action times
        
    return instructed_action # Returns the array of instructed actions at associated epochs

def crop_mi_inputs(features, truth_labels, included_truth_labels): # --- MAY BE DOABLE WITH NP.ARRAY SIMPLIFICATION!!!
    '''crop_mi_inputs
    '''
    bool_index = [] # Initialize an empty list to store the boolean indexes
    for truth in truth_labels: # Create the boolean indexer
        bool_index.append(truth in included_truth_labels) # Append either T/F is the current action is in the list to crop
    bool_index = bool_index
    # I have mixed feelings on this approach, on the one hand using the in comparison feels pythonic, on the other hand the lack of good direct indexing with strings feels clunky
    
    kept_labels = np.array(truth_labels)[bool_index] # Crop the labels to keep using boolean indexing // Labels converted to an array becasue apparently base Python doesn't have a reasonable way to deal with boolean indexing a list with a list
    kept_features = features[bool_index,:] # Crop the features to keep using boolean indexing
    
    return kept_labels, kept_features # Return the cropped arrays

# %% Part 2 Method Calls
'''
emg_data, emg_time = load_data("Smiechowski") # Load in data
epoched_data = epoch_data(emg_data, 500, 1) # Epoch the data
features, feature_shorthands = extract_features(epoched_data) # Extract the features and the shorthands // seriously why the shorthands?
instructed_action = make_truth_data(['rest','rock','rest','paper','rest','scissors']*10, 1) # Create the truth labels
truth_labels_ps, features_ps = crop_mi_inputs(features, instructed_action, ['paper','scissors']) # Crop out only the paper and scissors labels and features
'''
emg_data, emg_time = load_data("Klinger") # Load in data
epoched_data = epoch_data(emg_data, 500, 1) # Epoch the data
features, feature_shorthands = extract_features(epoched_data) # Extract the features and the shorthands // seriously why the shorthands?
instructed_action = make_truth_data(['rest','rock','rest','paper','rest','scissors']*10, 1) # Create the truth labels
truth_labels_ps, features_ps = crop_mi_inputs(features, instructed_action, ['paper','scissors']) # Crop out only the paper and scissors labels and features

# %% Part 3

def scatter_plot(two_features, feature_names, truth_matrix):
    '''scatter_plot
    '''
    # Two features is split into 4 variables for ease of use and readability // It's 2021, memory usage is a distant memory, seriously though back when I was all mCU all the time I had an optimization obsession
    # a features
    feature1a = two_features[:,0][truth_matrix == truth_matrix[0]] # The first feature
    feature2a = two_features[:,1][truth_matrix == truth_matrix[0]] # The second feature
    # b features
    feature1b = two_features[:,0][truth_matrix == truth_matrix[1]] # The first feature
    feature2b = two_features[:,1][truth_matrix == truth_matrix[1]] # The second feature
    
    # Plot the features
    plt.title(truth_matrix[0] + ' vs. ' + truth_matrix[1] + ' Scatter Plot')
    plt.xlabel(feature_names[0])
    plt.ylabel(feature_names[1])
    plt.scatter(feature1a, feature2a, label=truth_matrix[0])
    plt.scatter(feature1b, feature2b, label=truth_matrix[1])
    plt.legend()

plt.clf() # Clear the figure just in case

# Create a set of subplots comparing features
plt.suptitle('Feature comparisons') # Add a title over the whole figure
# Subplot Variance on channel 1 vs channel 2
plt.subplot(2,2,1)
scatter_plot(features_ps[:,1:3],feature_shorthands[1:3],truth_labels_ps)
# Subplot MAC vs ZC on channel 0
plt.subplot(2,2,2)
scatter_plot(np.transpose([features_ps[:,3],features_ps[:,6]]),[feature_shorthands[3],feature_shorthands[6]],truth_labels_ps)
# Subplot Variance on channel 0 vs ZC on channel 2
plt.subplot(2,2,3)
scatter_plot(np.transpose([features_ps[:,0],features_ps[:,8]]),[feature_shorthands[0],feature_shorthands[8]],truth_labels_ps)
# Use a tight layout to avoid plot overlap
plt.tight_layout()

# Save the current figure
plt.savefig('Paper_vs._Scissors_Scatter_Plot.png')

# %% Part 4



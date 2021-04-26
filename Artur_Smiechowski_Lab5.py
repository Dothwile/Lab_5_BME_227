"""
@author: Artur Smiechowski
A program that loads up emg data, epochs it, and extracts and calculates relevant data features.
Then it uses it to create a SVC model to predict future emg data. Then it evaluates the accuracy of the model both numerically
and graphically, saving those resulting plots locally.
"""
# %% Imports
import numpy as np
import matplotlib.pyplot as plt
import sklearn as skl
from sklearn import svm as svm # Otherwise get missing attribute error? Despite importing the entire module above?
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
    Parameters-
    name : str : The name associated with the loaded files
    Returns
    emg_data : 2D np array : the emg sensor data
    emg_time : 1D np array : the timestamps of associated samples
    
    loads both emg and time data of specified name
    '''
    emg_data = np.load(str(name) + '_RPS_Data.npy') # Load in data
    emg_time = np.load(str(name) + '_RPS_Time.npy') # Load in time
    
    return emg_data, emg_time # Return loaded data and time

def epoch_data(emg_data, fs, epoch_duration):
    '''epoch_data
    Parameters
    emg_data : 2D np array : the emg sensor data
    fs : int : the sampling frequency
    epoch_duration : float : the length of each epoch in seconds
    Returns
    epoched_data : 3D np array : the emg data divided into epochs
    
    Seperates the given emg data into epochs
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
    Parameters
    epoched_data : 3D np array : the emg data split into epochs
    Returns
    features : 2D np array : several calculated parameters of the epoched data
    feature_shorthands : 1D str list : a list of the feature name shorthands
    
    Exctracts and calculates a set of features for use in ML // Variance, Mean Absolute Vale (MAV), Zero Crossing events (ZC)
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
    features = skl.preprocessing.scale(features)
    #--- Cannot get mean and std exactly 0 and 1. Tested on smaller arrays with same syntax and was succesful
    #--- Output mean and std where X*10^-17 and 0.99999... respectively, assuming floating point error or issue with big array
    
    # Create the shorthand list
    feature_shorthands = ['Var_ch0','Var_ch1','Var_ch2','MAV_ch0','MAV_ch1','MAV_ch2','ZC_ch0','ZC_ch1','ZC_ch2'] # Why is this even in this method?
    
    return features, feature_shorthands # Return the normalized feature array

def make_truth_data(action_sequence, epochs_per_action):
    '''make_truth_data
    Parameters
    action_sequence : 1D str list : the whole seuqence of actions over the recording period
    epochs_per_action : int : the number of epochs per second
    Returns
    instructed_action : 1D str list : the total list of actions acounting for duplicate actions each second due to multipole epochs per second
    
    Creates a list of true actions factoring in repeats due to multiple epochs per second
    '''
    instructed_action = [] # Create the empty list for instructed actions
    for action in range(len(action_sequence)): # I guess we're assuming non-fractional values at this point? (I'd code around that but time)
        for epoch in range(epochs_per_action): # This could probably be done more elegantly but...
            instructed_action.append(action_sequence[action]) # Append the current action epochs_per_action times
        
    return instructed_action # Returns the array of instructed actions at associated epochs

def crop_mi_inputs(features, truth_labels, included_truth_labels): # --- MAY BE DOABLE WITH NP.ARRAY SIMPLIFICATION!!!
    '''crop_mi_inputs
    Parameters
    features : 2D np array : array of features from emg data
    truth_labels : 1D str list : the labels of actions that actually where performed at the associated epoch
    included_truch_labels : 1D str list : the actions to keep when cropping labels
    Returns
    kept_labels : 1D str list : the list of labels kept after cropping, in same order and count as before
    kept_features : 2D np array : the features associated with the kept labels, kept after cropping
    
    Crops out and returns truth labels and features that are specified
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

emg_data, emg_time = load_data("Smiechowski") # Load in data
epoched_data = epoch_data(emg_data, 500, 1) # Epoch the data
features, feature_shorthands = extract_features(epoched_data) # Extract the features and the shorthands // seriously why the shorthands?
instructed_action = make_truth_data(['rest','rock','rest','paper','rest','scissors']*10, 1) # Create the truth labels
truth_labels_ps, features_ps = crop_mi_inputs(features, instructed_action, ['paper','scissors']) # Crop out only the paper and scissors labels and features

''' //Used her data to verfiy/sanity check myself before. Turns out there's some really weird but rare anomalies in my data (see the 1 instance of the emg reading over 2100 (index 48905 of flattened array))
emg_data, emg_time = load_data("Klinger") # Load in data
epoched_data = epoch_data(emg_data, 500, 1) # Epoch the data
features, feature_shorthands = extract_features(epoched_data) # Extract the features and the shorthands // seriously why the shorthands?
instructed_action = make_truth_data(['rest','rock','rest','paper','rest','scissors']*10, 1) # Create the truth labels
truth_labels_ps, features_ps = crop_mi_inputs(features, instructed_action, ['paper','scissors']) # Crop out only the paper and scissors labels and features
'''

# %% Part 3

def scatter_plot(two_features, feature_names, truth_matrix):
    '''scatter_plot
    Parameters
    two_features : 2D np array : 2 select columns of features from the feature array
    feature_names : 1D str list : the names of the assocuated features
    truth_matrix : 1D str list : list of labels to use in plot
    Returns
    NONE
    
    Plots a scatter plot of the given features vs one another considering the action associated with them
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
scatter_plot(np.transpose([features_ps[:,1],features_ps[:,2]]),[feature_shorthands[1],feature_shorthands[2]],truth_labels_ps)
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

'''Explanation:
    It would seem that the greatest correlation is between Variances on channels 1 and 2, relatively falling along the same line.
    This may be due to the fact that this is the only plot in which the vlaues are of the same type (Variance), since similarly contracting muscles should
    produce simialr behavoir along the same parameters it could be explained that since paper and scissors are very similar gestures that would result in this correlation
    due to a similar ensemble of muscle contracting to produce them, ie same muscle contractions should produce a similar variance pattern
'''

# %% Part 4

def fit_classifier(features, labels):
    '''fit_classifier
    Parameters
    features : 2D np array : array of calculated features
    labels : 1D str list : labels to compare when binarizing
    Returns
    LinSVC : LinearSVC : a linear SVC model fit to the given feature and labels parameters
    
    Creates and fits a Linear SVC model to a given set of features and labels
    '''
    binary_labels = [] # Create an empty list to be the binarized lables // Not reassigning labels values to avoid recall issues
    for label in labels: # Iterate through each label in labels
        if label == 'scissors': # If scissors append a 1 to binary_labels, else must be paper so append -1
            binary_labels.append(1)
        else:
            binary_labels.append(-1)
            
    LinSVC = svm.LinearSVC(C=1e6) # Create the linear SVC object
    # Note: Due to small smaple size here (20 as instructed by lab) the model does not converge, meaning that subsequenct runs produce varying solutions and accuracies
    # ranging from 0.45 to 0.65 observed (I get the feeling this should be higher given the overfitting og the model, maybe because my own data is anomalous)
    
    return LinSVC.fit(features, binary_labels) # Return the trained classifier object // Note that training on 20 samples is not enough to converge

Paper_Scissors_LinSVC = fit_classifier(features_ps, truth_labels_ps) # Call the method to create a Paper vs Scissors SVM

w = np.ravel(Paper_Scissors_LinSVC.coef_) # Extract feature weights // Flattened since is already 1 dimensional
b = Paper_Scissors_LinSVC.intercept_[0] # Extract the bias // Pulled out of array since it's a single value

# Print the equation for z'
to_print = '' # The string to print, will be added to
for feature in range(len(feature_shorthands)): # Using a loop for this so it's actually readable
    rounded = round(w[feature], 3) # Round to 3 places
    to_print += f'{rounded} * {feature_shorthands[feature]} + ' # Add to to_print

print("z' = " + to_print + f" {round(b, 3)}") # Print the full equation
   
# %% Part 5

def predictor_histogram(trained_classifier, features, truth_labels):
    '''predictor_histogram
    Parameters
    trained_classifier : LinearSVC : a linear SVC model to create predictions
    features : 2D np array : emg features
    truth_labels : 1D str list : the list of truth labels
    Returns
    NONE
    
    Plots a histogram of the predicotrs calcuated by the given model
    '''
    # Isolate the z' scores of the actions
    paper_scores = trained_classifier.decision_function(features)[truth_labels == 'paper'] # Isolate the z' scores of the true paper actions
    scissors_scores = trained_classifier.decision_function(features)[truth_labels == 'scissors'] # Isolate the z' scores of the true scissors actions
    
    # Clear the figure // If you don't it just overlays the previous ones
    plt.clf()
    
    # Add axis labels and title for plot
    plt.title('Predictor Histogram')
    plt.xlabel('Predictor')
    plt.ylabel('Predicted Count')
    
    # Plot the histogram and add a legend
    plt.hist(paper_scores,bins=10, alpha=0.5, label="Predicted Paper")
    plt.hist(scissors_scores,bins=10, alpha=0.5, label="Predicted Scissors")
    plt.axvline(x=0, label="Threshhold") # Creates a vertical line at the threshhold
    plt.legend() # Enables data legend
    
    # Save the figure
    plt.savefig('Predictor_Histogram.png')
 
def evaluate_classifier(trained_classifier, features, truth_labels):
    '''evaluate_classifier
    Parameters
    trained_classifier : LinearSVC : a trained linear SVC model to evaluate
    features : 2D np array : the feature array to use in evaluation
    truth_labels : 1D str list : the list of truth labels
    Returns
    NONE
    
    Evaluates the accuracy of the the given SVC model, plots a confusion matrix of it too
    '''
    # Reusing binarization code since can't compare strings and floats // Feels like there's a better way since I'm repeating here but other than saving binary_labels as global var am unsure
    binary_labels = [] # Create an empty list to be the binarized lables // Not reassigning labels values to avoid recall issues
    for label in truth_labels: # Iterate through each label in labels
        if label == 'scissors': # If scissors append a 1 to binary_labels, else must be paper so append -1
            binary_labels.append(1)
        else:
            binary_labels.append(-1)

    print("Accuracy " + str(trained_classifier.score(features, binary_labels))) # Print the accuracy of the SVC
    
    
    plt.clf() # Clear figure just in case
    # Create the confusion matrix
    skl.metrics.plot_confusion_matrix(trained_classifier, features_ps, binary_labels)
    plt.savefig('Confusion_Matrix.png') # Save the figure

predictor_histogram(Paper_Scissors_LinSVC, features_ps, truth_labels_ps)
evaluate_classifier(Paper_Scissors_LinSVC, features_ps, truth_labels_ps)

# %% Part 6

# Use the same method calls as before but with lab partner
emg_data, emg_time = load_data("Klinger") # Load in data
epoched_data = epoch_data(emg_data, 500, 1) # Epoch the data
features, feature_shorthands = extract_features(epoched_data) # Extract the features and the shorthands // seriously why the shorthands?
instructed_action = make_truth_data(['rest','rock','rest','paper','rest','scissors']*10, 1) # Create the truth labels
truth_labels_ps, features_ps = crop_mi_inputs(features, instructed_action, ['paper','scissors']) # Crop out only the paper and scissors labels and features

Paper_Scissors_LinSVC = fit_classifier(features_ps, truth_labels_ps) # fit to her data, commented out when evaluating my model on her data

# Create the plots to evaluate the model
plt.clf() # Clear the figure just in case

# Create a set of subplots comparing features
plt.suptitle('Feature comparisons') # Add a title over the whole figure
# Subplot Variance on channel 1 vs channel 2
plt.subplot(2,2,1)
scatter_plot(np.transpose([features_ps[:,1],features_ps[:,2]]),[feature_shorthands[1],feature_shorthands[2]],truth_labels_ps)
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

# The other 2 plotting methods
predictor_histogram(Paper_Scissors_LinSVC, features_ps, truth_labels_ps)
evaluate_classifier(Paper_Scissors_LinSVC, features_ps, truth_labels_ps)

'''Explanation:
    My classifier does generally perform better on my own data than on Klinger's, but not as much as would be expected for an overfit classifier using the same data to train as it is predicting
    I understand that given proper training of a model (here 20 values is not enough) the accuracy should be higher (60% is susiciously low for an overfitted SVM), I think some of this
    is due to the weirdness of my own data. Normally I shuld expect some drop in accuracy for my partner's data since there are differences in the behavior of our data,
    ie the placement of her sensors could be different, our physiology different leading to different readings
    I can confirm that trained on her data even 20 entries perform much better when evaluating her data, minimum 70% accuracy, I chalking some if it up to issues with my data (need to check it before taking off sensors next time)
    Now this kind of training is not actually good for a real life application as you never want to train a model on the same values it is going to work with, if you do you get overfitting and loss
    of accuracy in a real world model
'''

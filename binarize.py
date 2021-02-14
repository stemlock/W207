import numpy as np

# Define a function that applies a threshold to turn real valued features into 0/1 features.
# 0 will mean "lower than mean" and 1 will mean "greater or equal to mean".

def binarize_continuous(data, thresholds=[]):
    
    '''
    A function that takes as input a set of data and converts the real valued features
    into 0/1 binary features based on the mean of the values for that feature.
    '''
    
    # Calculate the thresholds for each feature if it was not passed in as an argument
    if not thresholds:
        for i in range(data.shape[1]):
            thresholds.append(data[:,i].mean())
    
    # Initialize a new feature array with the same shape as the original data.
    binarized_data = np.empty(data.shape)

    # Apply a threshold  to each feature and create a true/false (1/0) matrix
    for feature in range(data.shape[1]):
        binarized_data[:,feature] = data[:,feature] > thresholds[feature]
        
    return binarized_data, thresholds
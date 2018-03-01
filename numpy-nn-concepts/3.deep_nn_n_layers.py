'''
Toy DNN to classify images from Simpsons dataset: img has Homer(0) or Margie(1)?
Dataset comes from Kaggle datasets (I just got Homer and Margie images)
I selected only images with format 480 X 320 and then resized it to 120 x 60
'''

import time, sys, os
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
from sklearn.model_selection import train_test_split
from PIL import Image
from scipy import ndimage
from _utils_dnn import *

def load_data(path):

	X = []
	Y = []

	idx_classes = 0
	idx_files = 0
	for folder_name in os.listdir(path):

		for filename in os.listdir(path+"/"+folder_name):

			img = Image.open(path+"/"+folder_name+"/"+filename)
			percent = 0.25
			img = img.resize( [int(percent * s) for s in img.size] )

			img = np.asarray(img)

			if img.shape[0] == 120 and img.shape[1] == 80:
				X.append(img)
				Y.append(idx_classes)

		idx_classes += 1

	X = np.array(X)
	Y = np.array(Y)

	# reshape to get (m, n_x) and (m, n_y)
	X = X.reshape(-1, 120 * 80 * 3) / 255
	Y = Y.reshape(-1, 1)

	return X, Y

X, Y = load_data("./data/dataset_simpsons")
# test_img = X[200]
# plt.imshow(test_img)
# plt.show()

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# transpose to get (n_x, m) and (n_y, m)
X_train = X_train.T
Y_train = Y_train.T
X_test = X_test.T
Y_test = Y_test.T

print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)

### DNN architecture
layers_dims = [X_train.shape[0], 20, 7, 5, 1] #  5-layer model

def model(X, Y, layers_dims, learning_rate=0.001, num_iterations=3000, print_cost=False):
    
    costs = []
    
    # Parameters initialization
    parameters = initialize_parameters_deep(layers_dims)
    
    # Loop (gradient descent)
    for i in range(0, num_iterations):

        # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID
        AL, caches = L_model_forward(X, parameters)
        
        # Compute cost
        cost = compute_cost(AL, Y)
    
        # Backward propagation
        grads = L_model_backward(AL, Y, caches)
 
        # Update parameters
        parameters = update_parameters(parameters, grads, learning_rate)
                
        # Print the cost every 100 training example
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" % (i, cost))
        if print_cost and i % 100 == 0:
            costs.append(cost)
            
    # plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    
    return parameters

parameters = model(X_train, Y_train, layers_dims, num_iterations = 1000, print_cost = True)

pred_train = predict(X_train, Y_train, parameters)

pred_test = predict(X_test, Y_test, parameters)

# print_mislabeled_images(classes, test_x, test_y, pred_test)

#### RESULTS
'''
(452, 28800)
(452, 1)
(113, 28800)
(113, 1)

Cost after iteration 0: 0.696929
Cost after iteration 100: 0.437315
Cost after iteration 200: 0.326806
Cost after iteration 300: 0.266513
Cost after iteration 400: 0.228604
Cost after iteration 500: 0.201916
Cost after iteration 600: 0.181477
Cost after iteration 700: 0.164853
Cost after iteration 800: 0.150721
Cost after iteration 900: 0.138403

Accuracy train_set: 0.96017699115
Accuracy test_set: 0.929203539823

'''








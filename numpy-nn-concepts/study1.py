import numpy as np
import cv2
from pprint import pprint

# load imagem as numpy array
def loadImage(path):
    image = cv2.imread(path)
    return image

# convert image with shape (w, h, c) to vector shape (w * h * c, 1)
def image2vector(image):
    v = image.reshape(image.shape[0] * image.shape[1] * image.shape[2], 1)
    return v

def normalizeImageVector(image_vector):
    norm = img_vector / 255
    return norm

# dimensions for 1 hidden nn
def nnLayers(n_in, n_hidden_neurons, n_out):
    n_x = n_in
    n_h = n_hidden
    n_y = n_out

    return n_x, n_h, n_y

def initialize_params(n_x, n_h, n_y):
    W1 = np.random.randn(n_h, n_x) / np.sqrt(n_x)
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h) / np.sqrt(n_h)
    b2 = np.zeros((n_y, 1))

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    
    return parameters

# relu activation function
def relu(Z):
    A = np.maximum(0,Z)
    # cache = Z 
    return A

# sigmoid activation function
def sigmoid(Z):
    A = 1/(1+np.exp(-Z))
    # cache = Z
    return A


def forward_propagation(X, parameters):
    # get params
    W1 = parameters.get('W1')
    b1 = parameters.get('b1')
    W2 = parameters.get('W2')
    b2 = parameters.get('b2')

    # forward calculations
    Z1 = np.dot(W1, X) + b1
    A1 = relu(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)

    # cache the values
    cache = {"Z1": Z1,
             "A1": A1,
             "Z2": Z2,
             "A2": A2}
    
    return A2, cache

# cross entropy cost
def compute_cost(A2, Y, parameters):

    m = Y.shape[1]
    logprobs = np.multiply(np.log(A2), Y)
    print("logprobs: ", logprobs)

    cost = -(np.sum(logprobs + (1-Y)*np.log(1-A2)))/m
    cost = np.squeeze(cost)

    return cost

def backward_propagation(parameters, cache, X, Y):

    m = X.shape[1]

    W1 = parameters.get('W1')
    W2 = parameters.get('W2')

    A1 = cache.get('A1')
    A2 = cache.get('A2')

    dZ2 = A2 - Y
    dW2 = np.dot(dZ2, A1.T) / m
    db2 = np.sum(dZ2, axis = 1, keepdims = True) / m
    dZ1 = W2.T * dZ2 * (1 - np.power(A1, 2))
    dW1 = np.dot(dZ1, X.T) / m
    db1 = np.sum(dZ1, axis = 1, keepdims = True) / m

    grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2}
    
    return grads



def model(X, layer_dims, learning_rate = 0.001, num_iterations = 10):

    grads = {}
    costs = []
    m = X.shape[1]
    (n_x, n_h, n_y) = layer_dims
    Y = X

    parameters = initialize_params(n_x, n_h, n_y)

    # Get W1, b1, W2 and b2 from the dictionary parameters.
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    print("m: ", m)
    print("X shape: ", X.shape)
    print("W1 shape: ", W1.shape)

    for i in range(0, num_iterations):

        A2, cache2 = forward_propagation(X, parameters)

        cost = compute_cost(A2, Y, parameters)

        dA2 = - (np.divide(Y, A2) - np.divide(1 - Y, 1 - A2))

        dA1, dW2, db2 = backward(dA2, cache2, 'sigmoid')
        dA0, dW1, db1 = linear_activation_backward(dA1, cache1, 'relu')

        grads['dW1'] = dW1
        grads['db1'] = db1
        grads['dW2'] = dW2
        grads['db2'] = db2

        parameters = update_parameters(parameters, grads, learning_rate)

        W1 = parameters["W1"]
        b1 = parameters["b1"]
        W2 = parameters["W2"]
        b2 = parameters["b2"]

    return parameters

img = loadImage('ball1.png')
img_vector = image2vector(img)
norm_vector = normalizeImageVector(img_vector)
print(norm_vector.shape)

n_x = norm_vector.shape[0]
n_h = 100
n_y = n_x
layer_dims = (n_x, n_h, n_y)

X = np.array(norm_vector)
# X = X.reshape(1, -1, 1)
# print(X.shape)

model(X, layer_dims)










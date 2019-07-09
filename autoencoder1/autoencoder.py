import numpy as np
from keras.layers import Input, Dense
from keras.models import Model
from keras.datasets import mnist
import matplotlib.pyplot as plt

# size of encoded representation
encoding_dim = 32

# input placeholder
input_img = Input(shape=(784,))

# encoded representation of the input
encoded = Dense(encoding_dim, activation='relu')(input_img)

# decode is the lossy reconstruction of the input
decoded = Dense(784, activation='sigmoid')(encoded)

# maps  an input to its reconstruction
autoencoder = Model(input_img, decoded)

# maps an input to its encoded representation
encoder = Model(input_img, encoded)

# placeholder for an encoded 32-dimensional input
encoded_input = Input(shape=(encoding_dim,))

# retrieve the last layer of the autoencoded model
decoded_layer = autoencoder.layers[-1]

# decoder model
decoder = Model(encoded_input, decoded_layer(encoded_input))

# configure the model to use a per-pixel binary crossentropy loss
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')


# --- preparint the input data
(x_train, _), (x_test, _) = mnist.load_data()

# normalize all values between 0 and 1
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

x_train = x_train.reshape(len(x_train), 784)
x_test = x_test.reshape(len(x_test), 784)

print(x_train.shape)
print(x_test.shape)


# --- training
autoencoder.fit(x_train, x_train,
                epochs=3,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, x_test))
encoded_images = encoder.predict(x_test)
decoded_images = decoder.predict(encoded_images)

# --- visualization
n = 20
plt.figure(figsize=(20, 4))
for i in range(n):
    # original images
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i+1+n)
    plt.imshow(decoded_images[i].reshape(28, 28))
    plt.gray()
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    plt.savefig('decoded.png', bbox_inches='tight')

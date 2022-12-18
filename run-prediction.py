# import keras
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# import os
import numpy as np
import matplotlib.pyplot as plt
import random


# load cifar10
# to use images from the test dataset for prediction
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)
print(type(x_test))
print(type(y_test[0]))

# cifar10 category label name
cifar10_labels = np.array([
    'airplane',
    'automobile',
    'bird',
    'cat',
    'deer',
    'dog',
    'frog',
    'horse',
    'ship',
    'truck'])


# load latest checkpoint
model = load_model('checkpoints/weights-008-1.0783.hdf5')

# prepare test image    
def convertCIFER10Data(image):
    img = image.astype('float32')
    c = np.zeros(32*32*3).reshape((1,32,32,3))
    c[0] = img
    return c

plt.figure(figsize=(16,16))

right = 0
mistake = 0

# run prediction
for i in range(75):
    # Random test image
    index = random.randint(0, x_test.shape[0])
    image = x_test[index]
    data = convertCIFER10Data(image)

    plt.subplot(10, 10, i+1)
    plt.tight_layout()
    plt.imshow(image)
    plt.axis('off')

    ret = model.predict(data, batch_size=1)

    bestnum = 0.0
    bestclass = 0
    for n in [0,1,2,3,4,5,6,7,8,9]:
        if bestnum < ret[0][n]:
            bestnum = ret[0][n]
            bestclass = n

    if y_test[index] == bestclass:
        plt.title(cifar10_labels[bestclass], fontsize=10)
        right += 1
    else:
        plt.title(cifar10_labels[bestclass] + "!=" + cifar10_labels[y_test[index][0]], color='#ff0000', fontsize=10)
        mistake += 1
                                                                   
plt.show()
print("Correct predictions:", right)
print("False predictions:", mistake)
print("Rate:", right/(mistake + right)*100, '%')
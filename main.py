import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns
from sklearn.metrics import confusion_matrix


# download cifar-10 image dataset
(X_train, y_train) , (X_test, y_test) = tf.keras.datasets.cifar10.load_data()
# x -> image and y -> label
## cifar10_labels = ['0=airplane','1=automobile','2=bird','3=cat','4=deer','5=dog','6=frog','7=horse','8=ship','9=truck']

# visualize dataset
## prepare a 4x4 grid
W_grid = 4
L_grid = 4
## prepare plot
fig, axes = plt.subplots(L_grid, W_grid, figsize = (15, 15))
axes = axes.ravel()
## get number of training images
n_training = len(X_train)

for i in np.arange(0, L_grid * W_grid):
    ## pick a random number
    index = np.random.randint(0, n_training)
    ## show image with index [index]
    axes[i].imshow(X_train[index])
    ## set title to label of selected image
    axes[i].set_title(y_train[index])
    axes[i].axis('off')
    
plt.subplots_adjust(hspace = 0.4)
## show random set of training images plus labels
### see ./tf-cifar_01.png
plt.show()



# prepare data
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

## change representation of categories
## number of categories expected
categories = 10
### cat 0 => 1,0,0,0,0,0,0,0,0,0
### cat 9 => 0,0,0,0,0,0,0,0,0,1
y_train = tf.keras.utils.to_categorical(y_train, categories)
y_test = tf.keras.utils.to_categorical(y_test, categories)

## normalize by 255
X_train = X_train/255
X_test = X_test/255

# X_train.shape => 50000 images, 32x32 pixel, 3 colour channels)
# input_shape => (32,32,3)
input_shape = X_train.shape[1:]

# training model
cnn = tf.keras.Sequential()
## 1st convolution layer / feature detection
cnn.add(tf.keras.layers.Conv2D(32, (3,3), activation = 'relu', input_shape = input_shape))
cnn.add(tf.keras.layers.Conv2D(32, (3,3), activation = 'relu'))
cnn.add(tf.keras.layers.MaxPooling2D(2,2))
cnn.add(tf.keras.layers.Dropout(0.3))

## 2nd convolution layer / feature detection
cnn.add(tf.keras.layers.Conv2D(64, (3,3), activation = 'relu'))
cnn.add(tf.keras.layers.Conv2D(64, (3,3), activation = 'relu'))
cnn.add(tf.keras.layers.MaxPooling2D(2,2))
cnn.add(tf.keras.layers.Dropout(0.3))

## flatten output
cnn.add(tf.keras.layers.Flatten())

## dense layers
cnn.add(tf.keras.layers.Dense(1024, activation = 'relu'))
cnn.add(tf.keras.layers.Dropout(0.3))

cnn.add(tf.keras.layers.Dense(1024, activation = 'relu'))

cnn.add(tf.keras.layers.Dense(10, activation = 'softmax'))
cnn.summary()

# compile model
## using RMSprop optimizer with accuracy metric
cnn.compile(optimizer = tf.keras.optimizers.RMSprop(1e-4), loss ='categorical_crossentropy', metrics =['accuracy'])


# fit model
epochs = 100

history = cnn.fit(X_train, y_train, batch_size = 512, epochs = epochs)


# evaluate model
## show accuracy after test
evaluation = cnn.evaluate(X_test, y_test)
print('Test Accuracy: {}'.format(evaluation[1]))
## show predicted classes array
predict_x = cnn.predict(X_test) 
predicted_classes = np.argmax(predict_x,axis=1)
print('Predicted Classes: ', predicted_classes)
## bring test labels to same format
y_test = y_test.argmax(1)

# test results
## prepare 7x7 output plot
L = 7
W = 7
fig, axes = plt.subplots(L, W, figsize = (12, 12))
axes = axes.ravel()

## print 7x7 test results
for i in np.arange(0, L*W):
    axes[i].imshow(X_test[i])
    # title with prediction vs true label
    axes[i].set_title('Prediction = {}\n True = {}'.format(predicted_classes[i], y_test[i]))
    axes[i].axis('off')

fig.tight_layout()
## show set of test results
### see ./tf-cifar_02.png
plt.show()

# print confusion matrix
## y-axis = predicted / x-axis = true
cm = confusion_matrix(predicted_classes, y_test)
cm
plt.figure(figsize = (10, 10))
sns.heatmap(cm, annot = True)
## show confusion matrix
### see ./tf-cifar_03.png
plt.show()
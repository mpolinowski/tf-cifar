## Tensorflow Image Classifier

### CIFAR-10

The [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images:


![Tensorflow Image Classifier](./tf-cifar_01.png)

> cifar10_labels = ['0=airplane','1=automobile','2=bird','3=cat','4=deer','5=dog','6=frog','7=horse','8=ship','9=truck']


The dataset is divided into five training batches and one test batch, each with 10000 images. The test batch contains exactly 1000 randomly-selected images from each class. The training batches contain the remaining images in random order, but some training batches may contain more images from one class than another. Between them, the training batches contain exactly 5000 images from each class. You can download the [Keras dataset](https://github.com/keras-team/keras/tree/master/keras/datasets) by:


```bash
from tensorflow.keras import datasets
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
```

## Run

```bash
python main.py
...
Epoch 100/100
98/98 [==============================] - 4s 41ms/step - loss: 0.5412 - accuracy: 0.8101
313/313 [==============================] - 1s 2ms/step - loss: 0.7025 - accuracy: 0.7602
Test Accuracy: 0.760200023651123
```

## Validation

Visually inspect the accuracy by printing random test predictions:

![Tensorflow Image Classifier](./tf-cifar_02.png)


### Confusion Matrix

> `y-axis = predicted / x-axis = true`

In a perfect model (accuracy=100%) we would expect a diagonal line from the top left to the bottom right. Every prediction that does not end up on this line is a false prediction:


![Tensorflow Image Classifier](./tf-cifar_03.png)


Most of the mistakes are based on falsely predicting:

* `4` (deer) in images of `2` (bird)
* `5` (dog) in images of `3` (cat)
* `3` (cat) in images of `5` (dog)


In words: The model cannot distinguish between cats and dogs and should be trained on those a little longer. While birds are sometimes mistaken for deers but not vice-versa.



## Transfer Learning

`train-tf-cifar-10-image-classifier.py` generates weights at checkpoints during a trainings run. These weights can be used to retrain the model for a different problem. Or use them in detections `run-prediction.py`.
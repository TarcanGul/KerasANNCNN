
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
import random
import logging


random.seed(1618)
np.random.seed(1618)
tf.set_random_seed(1618)

logger = tf.get_logger()
logger.setLevel(logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#ALGORITHM = "guesser"
#ALGORITHM = "tf_net"
ALGORITHM = "tf_conv"

#DATASET = "mnist_d"
#DATASET = "mnist_f"
#DATASET = "cifar_10"
#DATASET = "cifar_100_f"
DATASET = "cifar_100_c"

if DATASET == "mnist_d":
    NUM_CLASSES = 10
    IH = 28
    IW = 28
    IZ = 1
    IS = 784
elif DATASET == "mnist_f":
    NUM_CLASSES = 10
    IH = 28
    IW = 28
    IZ = 1
    IS = 784
elif DATASET == "cifar_10":
    NUM_CLASSES = 10
    IH = 32
    IW = 32
    IZ = 3
    IS = 32*32
elif DATASET == "cifar_100_f":
    NUM_CLASSES = 100
    IH = 32
    IW = 32
    IZ = 3
    IS = 32*32                                
elif DATASET == "cifar_100_c":
    NUM_CLASSES = 100
    IH = 32
    IW = 32
    IZ = 3
    IS = 32*32                                


#=========================<Classifier Functions>================================

def guesserClassifier(xTest):
    ans = []
    for entry in xTest:
        pred = [0] * NUM_CLASSES
        pred[random.randint(0, 9)] = 1
        ans.append(pred)
    return np.array(ans)

'''
Building an artificial neural net using keras. I used the number of epochs as 8 because I think it is a good balance for training.
I used categorical cross entropy for the loss function because it performs well on classification tasks.
I used two layers, with first layer having 128 neurons and the second one having NUM_CLASSES neurons.
For the first layer, I observed that having 2's power in the number of neurons helps with computation and accuracy.
The last layer has NUM_CLASSES neurons because at the end we want to get the info of which class the image belongs to.
'''
def buildTFNeuralNet(x, y, eps = 8):
    model = keras.Sequential()
    lossType = keras.losses.categorical_crossentropy
    opt = tf.train.AdamOptimizer()
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(128, activation="relu"))
    model.add(keras.layers.Dense(NUM_CLASSES, activation="softmax"))
    model.compile(optimizer = opt, loss = lossType)
    #Train model
    model.fit(x, y, epochs=eps, batch_size=100)
    return model

'''
Building an convolutional neural net using keras. I used the number of epochs as 10 because I think it is a good balance for training.
I used categorical cross entropy for the loss function because it performs well on classification tasks.
We use adam optimizer as a optimizer since that is what we used in class. It helps with optimizing the learning rate.
I started with two convolutional networks with ReLU's. The reason I did the second layer larger is that we have more 'shapes'
possible in second layer. In first layer, my imagination was we were only detecting lines or simple shapes. 
I set dropout to true because it will prevent overfitting.
The last two denses is similar to the ANN I built above.

I increased accuracy by making the second layer larger, using relu and increase the number of epochs.
'''
def buildTFConvNet(x, y, eps = 10, dropout = True, dropRate = 0.2):
    model = keras.Sequential()
    inShape = (IW,IH,IZ) 
    lossType = keras.losses.categorical_crossentropy
    opt = tf.train.AdamOptimizer()
    model.add(keras.layers.Conv2D(32, kernel_size = (3,3), activation="relu", input_shape = inShape))
    model.add(keras.layers.Conv2D(64, kernel_size = (3,3), activation="relu"))
    model.add(keras.layers.MaxPooling2D(pool_size = (2,2)))
    model.add(keras.layers.Flatten())
    if dropout:
        model.add(keras.layers.Dropout(dropRate))
    model.add(keras.layers.Dense(128, activation="relu"))
    model.add(keras.layers.Dense(NUM_CLASSES, activation="softmax"))
    model.compile(optimizer = opt, loss = lossType)
    #Train model
    model.fit(x, y, epochs=eps, batch_size=100)
    return model

#=========================<Pipeline Functions>==================================

def getRawData():
    if DATASET == "mnist_d":
        mnist = tf.keras.datasets.mnist
        (xTrain, yTrain), (xTest, yTest) = mnist.load_data()
    elif DATASET == "mnist_f":
        mnist = tf.keras.datasets.fashion_mnist
        (xTrain, yTrain), (xTest, yTest) = mnist.load_data()
    elif DATASET == "cifar_10":
        cifar = tf.keras.datasets.cifar10
        (xTrain, yTrain), (xTest, yTest) = cifar.load_data()
    elif DATASET == "cifar_100_f":
        cifar = tf.keras.datasets.cifar100
        (xTrain, yTrain), (xTest, yTest) = cifar.load_data(label_mode="fine")
    elif DATASET == "cifar_100_c":
        cifar = tf.keras.datasets.cifar100
        (xTrain, yTrain), (xTest, yTest) = cifar.load_data(label_mode="coarse")
    else:
        raise ValueError("Dataset not recognized.")
    print("Dataset: %s" % DATASET)
    print("Shape of xTrain dataset: %s." % str(xTrain.shape))
    print("Shape of yTrain dataset: %s." % str(yTrain.shape))
    print("Shape of xTest dataset: %s." % str(xTest.shape))
    print("Shape of yTest dataset: %s." % str(yTest.shape))
    return ((xTrain, yTrain), (xTest, yTest))

def normalizeData(raw):
    return ((raw[0][0] / 256, raw[0][1]), (raw[1][0] / 256, raw[1][1]))

def preprocessData(raw):
    ((xTrain, yTrain), (xTest, yTest)) = normalizeData(raw)
    xTrainP = xTrain.reshape((xTrain.shape[0], IH, IW, IZ))
    xTestP = xTest.reshape((xTest.shape[0], IH, IW, IZ))
    yTrainP = to_categorical(yTrain, NUM_CLASSES)
    yTestP = to_categorical(yTest, NUM_CLASSES)
    print("New shape of xTrain dataset: %s." % str(xTrainP.shape))
    print("New shape of xTest dataset: %s." % str(xTestP.shape))
    print("New shape of yTrain dataset: %s." % str(yTrainP.shape))
    print("New shape of yTest dataset: %s." % str(yTestP.shape))
    return ((xTrainP, yTrainP), (xTestP, yTestP))



def trainModel(data):
    xTrain, yTrain = data
    if ALGORITHM == "guesser":
        return None   # Guesser has no model, as it is just guessing.
    elif ALGORITHM == "tf_net":
        print("Building and training TF_NN.")
        return buildTFNeuralNet(xTrain, yTrain)
    elif ALGORITHM == "tf_conv":
        print("Building and training TF_CNN.")
        return buildTFConvNet(xTrain, yTrain)
    else:
        raise ValueError("Algorithm not recognized.")



def runModel(data, model):
    if ALGORITHM == "guesser":
        return guesserClassifier(data)
    elif ALGORITHM == "tf_net":
        print("Testing TF_NN.")
        preds = model.predict(data)
        for i in range(preds.shape[0]):
            oneHot = [0] * NUM_CLASSES
            oneHot[np.argmax(preds[i])] = 1
            preds[i] = oneHot
        return preds
    elif ALGORITHM == "tf_conv":
        print("Testing TF_CNN.")
        preds = model.predict(data)
        for i in range(preds.shape[0]):
            oneHot = [0] * NUM_CLASSES
            oneHot[np.argmax(preds[i])] = 1
            preds[i] = oneHot
        return preds
    else:
        raise ValueError("Algorithm not recognized.")



def evalResults(data, preds):
    xTest, yTest = data
    acc = 0
    for i in range(preds.shape[0]):
        if np.array_equal(preds[i], yTest[i]):   acc = acc + 1
    accuracy = acc / preds.shape[0]
    print("Classifier algorithm: %s" % ALGORITHM)
    print("Classifier accuracy: %f%%" % (accuracy * 100))



#=========================<Main>================================================

def main():
    raw = getRawData()
    data = preprocessData(raw)
    model = trainModel(data[0])
    preds = runModel(data[1][0], model)
    evalResults(data[1], preds)



if __name__ == '__main__':
    main()

import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
import sys
import pickle
import gzip
from tensorflow.keras.datasets import mnist
import tensorflow
import timeit

def load_mnist_dataset():
    f = gzip.open('mnist.pkl.gz', 'rb')
    if sys.version_info < (3,):
        data = pickle.load(f)
    else:
        data = pickle.load(f, encoding='bytes')
    f.close()

    return data

class tanh:
    @staticmethod
    def activation(x):
        y = (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
        return y

    @staticmethod
    def prime(x):
        y = 1 - (tanh.activation(x) ** 2)
        return y

class sigmoid:
    @staticmethod
    def activation(x):
        y = 1 / (1 + np.exp(-x))
        return y

    @staticmethod
    def prime(x):
        y = sigmoid.activation(x) * (1 - sigmoid.activation(x))
        return y

class relu:
    @staticmethod
    def activation(x):
        y = np.maximum(0, x)
        return y

    @staticmethod
    def prime(x):
        x[x <= 0] = 0
        x[x > 0] = 1
        return x

class softmax:
    @staticmethod
    def activation(x):
        exps = np.exp(x - np.max(x))
        return exps / np.sum(exps, axis=1).reshape(-1, 1)

class Initialization:

    @staticmethod
    def Zeros(input_channel, output_channel):
        w = np.zeros((input_channel, output_channel))
        sdw = np.zeros((input_channel, output_channel))
        vdw = np.zeros((input_channel, output_channel))

        b = np.zeros((1, output_channel))
        sdb = np.zeros((1, output_channel))
        vdb = np.zeros((1, output_channel))

        return w, b, sdw, sdb, vdw, vdb

    @staticmethod
    def Xavier(input_channel, output_channel):
        w = np.random.randn(input_channel, output_channel) * np.sqrt(1 / input_channel)
        sdw = np.zeros((input_channel, output_channel))
        vdw = np.zeros((input_channel, output_channel))

        b = np.zeros((1, output_channel))
        sdb = np.zeros((1, output_channel))
        vdb = np.zeros((1, output_channel))

        return w, b, sdw, sdb, vdw, vdb

    @staticmethod
    def He(input_channel, output_channel):
        w = np.random.randn(input_channel, output_channel) * np.sqrt(2 / input_channel)
        sdw = np.zeros((input_channel, output_channel))
        vdw = np.zeros((input_channel, output_channel))

        b = np.zeros((1, output_channel))
        sdb = np.zeros((1, output_channel))
        vdb = np.zeros((1, output_channel))

        return w, b, sdw, sdb, vdw, vdb

    @staticmethod
    def Kumar(input_channel, output_channel):
        w = np.random.randn(input_channel, output_channel) * np.sqrt(12.96 / input_channel)
        sdw = np.zeros((input_channel, output_channel))
        vdw = np.zeros((input_channel, output_channel))

        b = np.zeros((1, output_channel))
        sdb = np.zeros((1, output_channel))
        vdb = np.zeros((1, output_channel))

        return w, b, sdw, sdb, vdw, vdb

class optimizers():

    @staticmethod
    def GD(index, dw, db, learning_rate):
        model.layers[index].weight -= learning_rate * dw
        model.layers[index].bias -= learning_rate * db

    @staticmethod
    def RMSprop(index, gamma, dw, db, vdw, vdb, learning_rate):
        vdw = gamma * vdw + (1 - gamma) * dw ** 2
        vdb = gamma * vdb + (1 - gamma) * db ** 2

        model.layers[index].weight -= (learning_rate / (np.sqrt(vdw + 1e-08))) * dw
        model.layers[index].bias -= (learning_rate / (np.sqrt(vdb + 1e-08))) * db

    @staticmethod
    def Adam(index, gamma1, gamma2, dw, db, mdw, mdb, vdw, vdb, currentepoch, alpha):
        mdw = gamma1 * mdw + (1 - gamma1) * dw
        mdb = gamma1 * mdb + (1 - gamma1) * db

        vdw = gamma2 * vdw + (1 - gamma2) * dw ** 2
        vdb = gamma2 * vdb + (1 - gamma2) * db ** 2

        mdw_corr = mdw / (1 - np.power(gamma1, currentepoch + 1))
        mdb_corr = mdb / (1 - np.power(gamma1, currentepoch + 1))

        vdw_corr = vdw / (1 - np.power(gamma2, currentepoch + 1))
        vdb_corr = vdb / (1 - np.power(gamma2, currentepoch + 1))

        model.layers[index].weight -= (alpha / (np.sqrt(vdw_corr + 1e-08))) * mdw_corr
        model.layers[index].bias -= (alpha / (np.sqrt(vdb_corr + 1e-08))) * mdb_corr

    @staticmethod
    def Adamax(index, gamma1, gamma2, dw, db, vdw, vdb, sdw, sdb, currentepoch, learning_rate):
        vdw = gamma1 * vdw + (1 - gamma1) * dw
        vdb = gamma1 * vdb + (1 - gamma1) * db

        sdw = np.maximum(gamma2 * sdw, np.abs(dw))
        sdb = np.maximum(gamma2 * sdb, np.abs(db))

        vdw_corr = vdw / (1 - np.power(gamma1, currentepoch + 1))
        vdb_corr = vdb / (1 - np.power(gamma1, currentepoch + 1))

        model.layers[index].weight -= (learning_rate / (np.sqrt(sdw + 1e-08))) * vdw_corr
        model.layers[index].bias -= (learning_rate / (np.sqrt(sdb + 1e-08))) * vdb_corr

class layers():
    class Dense():
        def __init__(self, input_channel, output_channel, activation=None, initialization=None):

            self.initialization = initialization

            if self.initialization == 'Xavier':
                self.weight, self.bias, self.sdw, self.sdb, self.vdw, self.vdb = \
                    Initialization.Xavier(input_channel, output_channel)
            elif self.initialization == 'He':
                self.weight, self.bias, self.sdw, self.sdb, self.vdw, self.vdb = \
                    Initialization.He(input_channel, output_channel)
            elif self.initialization == 'Kumar':
                self.weight, self.bias, self.sdw, self.sdb, self.vdw, self.vdb = \
                    Initialization.Kumar(input_channel, output_channel)
            else:
                self.weight, self.bias, self.sdw, self.sdb, self.vdw, self.vdb = \
                    Initialization.Zeros(input_channel, output_channel)

            model.activations.append(activation)
            self.activation = activation

        def forward(self, X):
            self.X = X

            z = np.dot(self.X, self.weight) + self.bias
            return z

        def backward(self):
            return self.X.T

class CategoricalCrossEntropy():
    def __init__(self, z, y_true):
        self.y_predict = z
        self.y_true = y_true

    def forward(self):
        m = self.y_predict.shape[0]

        return (-1) * (1 / m) * np.sum((self.y_true * np.log(self.y_predict)))

    def backward(self):
        delta = self.y_predict - self.y_true
        return delta

class BinaryCrossEntropy:
    def __init__(self, z, y_true):
        self.y_predict = z
        self.y_true = y_true

    def forward(self):
        m = self.y_predict.shape[0]
        return (-1) * (1 / m) * (np.sum((self.y_true * np.log(self.y_predict + 1e-8)) +
                                        ((1 - self.y_true) * (np.log(1 - self.y_predict + 1e-8)))))

    def backward(self):
        delta = self.y_predict - self.y_true
        return delta

class sequential():

    def __init__(self):
        self.layers = []
        self.activations = []
        self.Loss_list = []
        self.epochs_list = []
        self.accuracy_list = []

    def add(self, layer):
        self.layers.append(layer)
        return layer

    def compile(self, loss, optimizer):
        self.cost = loss
        self.optimizer = optimizer

    def GDScheduler(self, lr):
        self.learning_rate = lr

    def fit(self, X_train, y_train, batch_size, epochs):

        global cost_function, cost, accuracy
        m = X_train.shape[0]
        _z = []
        _a = []
        for i in range(epochs):
            current_epoch = i
            start = timeit.default_timer()

            for p in range(m // batch_size):

                k = p * batch_size
                l = (p + 1) * batch_size
                a = X_train[k:l]
                y = y_train[k:l]
                # Feed forward
                for j, eachLayer in enumerate(model.layers):
                    z = eachLayer.forward(a)
                    _z.append(z)
                    a = eval(model.activations[j]).activation(z)
                    _a.append(a)                                    # Storing activation values for back propogation

                # Calculating cost function
                if self.cost == 'BinaryCrossEntropy':
                    cost_function = 'BinaryCrossEntropy'

                if self.cost == 'CategoricalCrossEntropy':
                    cost_function = 'CategoricalCrossEntropy'

                loss = eval(cost_function)(a, y)
                cost = loss.forward()

                # Backpropagation
                delta = a - y
                deliver_delta = delta
                for j in reversed(range(1, len(model.layers))):

                    dw = (1 / m) * np.dot(_a[j - 1].T, deliver_delta)
                    db = (1 / m) * np.sum(deliver_delta)

                    deliver_delta = np.dot(deliver_delta, model.layers[j].weight.T) * \
                                    eval(model.activations[j - 1]).prime(_z[j - 1])

                    # update parameters
                    if self.optimizer == 'GD':
                        optimizers.GD(j, dw, db, self.learning_rate)

                    if self.optimizer == 'RMSprop':
                        optimizers.RMSprop(j, 0.9, dw, db, model.layers[j].sdw, model.layers[j].sdb, self.learning_rate)

                    if self.optimizer == 'Adam':
                        optimizers.Adam(j, 0.9, 0.999, dw, db, model.layers[j].vdw, model.layers[j].vdb,
                                        model.layers[j].sdw, model.layers[j].sdb, current_epoch, self.learning_rate)

                    if self.optimizer == 'Adamax':
                        optimizers.Adamax(j, 0.9, 0.999, dw, db, model.layers[j].vdw, model.layers[j].vdb,
                                          model.layers[j].sdw, model.layers[j].sdb, current_epoch,
                                          self.learning_rate)

                accuracy = np.mean(np.equal(np.argmax(y, axis=-1), np.argmax(a, axis=-1))) * 100

            end = timeit.default_timer()

            print("epochs:" + str(i + 1) + " | " +
                  "runtime: {} s".format(float(round(end - start, 3))) + " | " +
                  "Loss:" + str(cost) + " | " +
                  "Accuracy: {} %".format(float(round(accuracy, 3))))

            if i % 10 == 0:
                self.accuracy_list.append(accuracy)
                self.Loss_list.append(cost)
                self.epochs_list.append(i)

        # accuracy Plot
        accuracy_list = np.array(self.accuracy_list)
        accuracy_list = accuracy_list.reshape(-1, 1)

        # Loss Plot
        loss_array = np.array(self.Loss_list)
        y_loss = loss_array.reshape(-1, 1)
        x_epochs = np.array(self.epochs_list).reshape(-1, 1)

        accuracy_data = pd.DataFrame()
        accuracy_data['0'] = x_epochs.reshape(1, -1)[0]
        accuracy_data['1'] = accuracy_list.reshape(1, -1)[0]
        accuracy_data.to_csv('accuracy.txt', index=False, header=False, sep=" ")

        loss_data = pd.DataFrame()
        loss_data['0'] = x_epochs.reshape(1, -1)[0]
        loss_data['1'] = y_loss.reshape(1, -1)[0]
        loss_data.to_csv('cost.txt', index=False, header=False, sep=" ")

        plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 1)
        plt.plot(x_epochs, accuracy_list)
        plt.xlabel('epochs')
        plt.ylabel('accuracy')
        plt.title('epochs_vs_accuracy')

        plt.subplot(1, 2, 2)
        plt.plot(x_epochs, y_loss)
        plt.xlabel('epochs')
        plt.ylabel('Loss')
        plt.title('epochs_vs_loss')

        fig1 = plt.gcf()
        plt.show()
        fig1.savefig('Results.png')

        print("Training accuracy: {} %".format(accuracy))

    @staticmethod
    def predict(x):

        a = x
        for j, eachLayer in enumerate(model.layers):
            z = eachLayer.forward(a)
            a = eval(model.activations[j]).activation(z)

        return a

    @staticmethod
    def confusion_matrix(data_array, labels):

        dim = len(data_array[0])
        cm = np.zeros((dim, dim), int)

        for i in range(len(data_array)):
            truth = np.argmax(data_array[i])
            predicted = np.argmax(labels[i])
            cm[truth, predicted] += 1

        plt.figure(figsize=(10, 7))
        sn.heatmap(cm, annot=True, fmt='d')
        plt.xlabel("Predicted")
        plt.ylabel("Truth")

        fig1 = plt.gcf()
        plt.show()
        fig1.savefig('cm.png')
        return cm

def evaluate(y_test, y_predicted):
    accuracy = np.mean(np.equal(np.argmax(y_test, axis=-1), np.argmax(y_predicted, axis=-1))) * 100
    print("Testing accuracy: {} %".format(accuracy))

if __name__ == '__main__':
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    X_train = X_train / 255
    X_test = X_test / 255

    y_train = tensorflow.keras.utils.to_categorical(y_train, 10).T
    y_test = tensorflow.keras.utils.to_categorical(y_test, 10).T

    X_train = X_train.reshape(-1, 28 * 28)
    X_test = X_test.reshape(-1, 28 * 28)
    y_train = y_train.T
    y_test = y_test.T

    model = sequential()

    model.add(layers.Dense(784, 50, activation='relu', initialization='He'))
    model.add(layers.Dense(50, 10, activation='softmax', initialization='He'))

    model.compile(loss='CategoricalCrossEntropy', optimizer='Adam')
    model.GDScheduler(lr=0.01)
    model.fit(X_train, y_train, batch_size=6000, epochs=100)

    y_predicted = model.predict(X_test)

    evaluate(y_test, y_predicted)
    model.confusion_matrix(y_test, y_predicted)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ray_tracer import n_rots


# Helper functions
def get_pred(A):
    return np.argmax(A, axis=0)

def get_acc(pred,Y):
    return np.sum(pred == Y) / Y.size

class NeuralNetwork:
    def __init__(self, layers, onehot=True):
        self.onehot = onehot
        self.layers = []
        self.weights = []
        self.biases = []
        for i,l in enumerate(layers[1:]):
            self.layers.append(0)
            self.weights.append(np.random.rand(l,layers[i])-0.5)
            self.biases.append(np.random.rand(l,1)-0.5)
    
    # Activation function
    def ReLU(self, Z):
        return np.maximum(Z,0)
    def deriv_ReLU(self, Z):
        return (Z > 0)
    def softmax(self, Z):
        return np.exp(Z) / sum(np.exp(Z))

    # One-hot encode the output data
    def one_hot(self, Y):
        one_hot_Y = np.zeros((Y.size, int(Y.max()) + 1))
        one_hot_Y[np.arange(Y.size), Y.astype('int')] = 1
        return one_hot_Y.T

    # Forward input
    def forward_prop(self, X):
        net = [X]
        for i,l in enumerate(self.layers):
            Z = self.weights[i].dot(net[-1]) + self.biases[i]
            net.append(Z)
            if i == len(self.layers)-1:
                net.append(self.ReLU(Z))
            else:
                net.append(self.softmax(Z))
        return net
    
    # Adjust weights&biases with back propagation
    def back_prop(self, net, Y, alpha=.1):
        m = Y.size
        if self.onehot:
            Y = self.one_hot(Y)
        dZ = [2*(net[-1] - Y)]
        dW = []; dB = []
        for i in [-(r+1) for r in range(len(self.layers))]:
            dW.append( dZ[-1].dot(net[2*i-1].T) / m )
            dB.append( np.sum(dZ[-1] / m) )
            if 2*i-2 > -len(net):
                dZ.append( self.weights[i].T.dot(dZ[-1]) * self.deriv_ReLU(net[2*i-2]) )
        dW.reverse(); dB.reverse()
        for i in range(len(dW)):
            self.weights[i] -= alpha*dW[i]
            self.biases[i] -= alpha*dB[i]
    
    # Train network
    def train(self, X, Y, iters, alpha=.1):
        acc_list = []
        for i in range(iters):
            net = self.forward_prop(X)
            self.back_prop(net, Y, alpha)
            if (i%50 == 0):
                acc_list.append( get_acc(get_pred(net[-1]),Y) )
                print(f"Iterations: {i}")
                print(f"Accuracy: {acc_list[-1]}")
        return acc_list



layers = [100,15,5] # Input, hidden layers, output
train_iterations = 5000
repeats = 1
ratio = 0.7

train_acc = []
test_acc = []
for tests in range(repeats):
    data = pd.read_csv("data.csv", header=None)
    data = np.array(data)
    m, n = data.shape # m samples, n attributes
    np.random.shuffle(data)

    x = int(ratio*m)
    data_train = data[:x].T
    Y_train = data_train[0]
    X_train = data_train[-n_rots:]
    X_train = (X_train-np.mean(X_train,axis=0))/np.std(X_train,axis=0)

    data_test = data[x:].T
    Y_test = data_test[0]
    X_test = data_test[-n_rots:]
    X_test = (X_test-np.mean(X_test,axis=0))/np.std(X_test,axis=0)

    N = NeuralNetwork(layers, onehot=True)
    accuracy = N.train(X_train, Y_train, train_iterations)
    pred = N.forward_prop(X_test)[-1]   
    train_acc.append(accuracy[-1])
    test_acc.append(get_acc(get_pred(pred),Y_test))
print(f"\nTrain data accuracy: {np.mean(np.array(train_acc))}")
print(f"\nTest data accuracy: {np.mean(np.array(test_acc))}")

plt.plot(np.array(accuracy)*100)
plt.show()
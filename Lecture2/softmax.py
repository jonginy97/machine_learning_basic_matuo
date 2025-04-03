import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
from tensorflow.keras.datasets import mnist

import matplotlib.pyplot as plt


def np_log(x):
    return np.log(np.clip(a=x, a_min=1e-10, a_max=1e+10))

def softmax(x):
    x -= x.max(axis=1, keepdims=True) # Prevent overflow in exp
    x_exp = np.exp(x)
    return x_exp / np.sum(x_exp, axis=1, keepdims=True)

(x_mnist_1, y_mnist_1), (x_mnist_2, y_mnist_2) = mnist.load_data()

# # Visualize the data
# plt.imshow(x_mnist_1[0], cmap='gray')
# plt.title('Label: {}'.format(y_mnist_1[0]))
# plt.axis('off')
# plt.show()



x_mnist = np.r_[x_mnist_1, x_mnist_2]
y_mnist = np.r_[y_mnist_1, y_mnist_2]

x_mnist = x_mnist.astype('float32') / 255.
y_mnist = np.eye(N=10)[y_mnist.astype('int32').flatten()]

x_mnist=x_mnist.reshape(x_mnist.shape[0],-1)

x_train_mnist, x_test_mnist, y_train_mnist, y_test_mnist = train_test_split(x_mnist, y_mnist, test_size=10000)
x_train_mnist, x_valid_mnist, y_train_mnist, y_valid_mnist = train_test_split(x_train_mnist, y_train_mnist, test_size=10000)

# Weights (input dimension: 784, output dimension: 10)
W_mnist = np.random.uniform(low=-0.08, high=0.08, size=(784, 10)).astype('float32')
b_mnist = np.zeros(shape=(10,)).astype('float32')

def train_mnist(x, y, eps=1.0):
    """
    :param x: np.ndarray, input data, shape=(batch_size, input dimension)
    :param y: np.ndarray, target labels, shape=(batch_size, output dimension)
    :param eps: float, learning rate
    """
    global W_mnist, b_mnist

    batch_size = x.shape[0]

    # Prediction
    y_hat = softmax(np.matmul(x, W_mnist) + b_mnist) # shape: (batch_size, output dimension)

    # Evaluate the objective function
    cost = (- y * np_log(y_hat)).sum(axis=1).mean()
    delta = y_hat - y # shape: (batch_size, output dimension)

    # Update parameters
    dW = np.matmul(x.T, delta) / batch_size # shape: (input dimension, output dimension)
    db = np.matmul(np.ones(shape=(batch_size,)), delta) / batch_size # shape: (output dimension,)
    W_mnist -= eps * dW
    b_mnist -= eps * db

    return cost

def valid_mnist(x, y):
    y_hat = softmax(np.matmul(x, W_mnist) + b_mnist)
    cost = (- y * np_log(y_hat)).sum(axis=1).mean()

    return cost, y_hat

for epoch in range(100):
    x_train_mnist, y_train_mnist = shuffle(x_train_mnist, y_train_mnist)
    cost = train_mnist(x_train_mnist, y_train_mnist)
    cost, y_pred = valid_mnist(x_valid_mnist, y_valid_mnist)
    if epoch % 10 == 9 or epoch == 0:
        print('EPOCH: {}, Valid Cost: {:.3f}, Valid Accuracy: {:.3f}'.format(
            epoch + 1,
            cost,
            accuracy_score(y_valid_mnist.argmax(axis=1), y_pred.argmax(axis=1))
        ))
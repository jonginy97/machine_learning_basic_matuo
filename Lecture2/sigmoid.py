from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from keras.datasets import mnist

import numpy as np

np.random.seed(34)


def sigmoid(x):
    # Simple implementation
    # return 1 / (1 + np.exp(-x))

    # Implementation with overflow prevention for exp
    # When x >= 0, sigmoid(x) = 1 / (1 + exp(-x))
    # When x < 0, sigmoid(x) = exp(x) / (1 + exp(x))
    return np.exp(np.minimum(x, 0)) / (1 + np.exp(- np.abs(x)))

# OR dataset
x_train_or = np.array([[0, 1], [1, 0], [0, 0], [1, 1]])
y_train_or = np.array([[1], [1], [0], [1]])
x_valid_or, y_valid_or = x_train_or, y_train_or
x_test_or, y_test_or = x_train_or, y_train_or

# Weights (input dimension: 2, output dimension: 1)
W_or = np.random.uniform(low=-0.08, high=0.08, size=(2, 1)).astype('float32')
b_or = np.zeros(shape=(1,)).astype('float32')

# Preventing log of zero
def np_log(x):
    return np.log(np.clip(a=x, a_min=1e-10, a_max=1e+10))

def train_or(x, y, eps=1.0):
    """
    :param x: np.ndarray, input data, shape=(batch_size, input dimension)
    :param y: np.ndarray, teacher labels, shape=(batch_size, output dimension)
    :param eps: float, learning rate
    """
    global W_or, b_or

    batch_size = x.shape[0]

    # Prediction
    y_hat = sigmoid(np.matmul(x, W_or) + b_or) # shape: (batch_size, output dimension)

    # Evaluation of the objective function
    cost = (- y * np_log(y_hat) - (1 - y) * np_log(1 - y_hat)).mean()
    delta = y_hat - y # shape: (batch_size, output dimension)

    # Parameter update
    dW = np.matmul(x.T, delta) / batch_size # shape: (input dimension, output dimension)
    db = np.matmul(np.ones(shape=(batch_size,)), delta) / batch_size # shape: (output dimension,)
    W_or -= eps * dW
    b_or -= eps * db

    return cost

def valid_or(x, y):
    y_hat = sigmoid(np.matmul(x, W_or) + b_or)
    cost = (- y * np_log(y_hat) - (1 - y) * np_log(1 - y_hat)).mean()
    return cost, y_hat

for epoch in range(1000):
    x_train_or, y_train_or = shuffle(x_train_or, y_train_or)
    cost = train_or(x_train_or, y_train_or)
    cost, y_pred = valid_or(x_valid_or, y_valid_or)

print(y_pred)


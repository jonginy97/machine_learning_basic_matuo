import os
import sys

import numpy as np
import pandas as pd

sys.modules['tensorflow'] = None

def load_fashionmnist():
    # Load training data
    x_train = np.load('data/x_train.npy')
    y_train = np.load('data/y_train.npy')
    
    # Load test data
    x_test = np.load('data/x_test.npy')
    
    x_train = x_train.reshape(-1, 784).astype('float32') / 255
    y_train = np.eye(10)[y_train.astype('int32')]
    x_test = x_test.reshape(-1, 784).astype('float32') / 255
    
    return x_train, y_train, x_test

x_train, y_train, x_test = load_fashionmnist()

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

def softmax(x):
    x -= np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# weights
W = np.random.uniform(-0.08, 0.08, (784, 10)).astype('float32')
b = np.zeros(10).astype('float32')

# Split training and validation data
x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.1)

def train(x, t, eps=0.1):
    global W, b
    batch_size = x.shape[0]
    y = softmax(np.dot(x, W) + b)
    delta = y - t
    dW = np.dot(x.T, delta) / batch_size
    db = np.sum(delta, axis=0) / batch_size
    W -= eps * dW
    b -= eps * db

def valid(x, t):
    y = softmax(np.dot(x, W) + b)
    return np.mean(np.argmax(y, axis=1) == np.argmax(t, axis=1))

for epoch in range(1):
    for i in range(len(x_train)):
        xi = x_train[i:i+1]
        ti = y_train[i:i+1]
        train(xi, ti)
    acc = valid(x_valid, y_valid)
    print(f"Epoch {epoch}: Validation Accuracy = {acc:.4f}")

y_test_pred = softmax(np.dot(x_test, W) + b)
y_pred = np.argmax(y_test_pred, axis=1)

# submission = pd.Series(y_pred, name='label')
# submission.to_csv('submission_pred.csv', header=True, index_label='id')
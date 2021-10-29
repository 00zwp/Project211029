import numpy as np
from keras.datasets import mnist

np.random.seed(666)

#数据集的导入

(trainX,trainY),(testX,testY)= mnist.load_data(path='./dataset/mnist.npz')

print(type(trainX))
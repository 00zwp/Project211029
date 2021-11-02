import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Dense, Dropout, Activation, Flatten
from keras.utils import np_utils
import keras
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
np.random.seed(666)

if __name__ == "__main__":
    #数据集的导入
    (trainX,trainY),(testX,testY)= mnist.load_data(path='./dataset/mnist.npz')
    trainX = trainX.reshape(-1, 28, 28, 1).astype(np.float32)
    testX = testX.reshape(-1, 28, 28, 1).astype(np.float32)
    trainY = np_utils.to_categorical(trainY, 10)  # to one_hot
    testY = np_utils.to_categorical(testY, 10)  # to one_hot

    trainX /= 255.
    testX /= 255.

    def model():
        model =Sequential()
        model.add(Flatten())
        model.add(Dense(1024,input_shape=(28,28,1)))
        model.add(Dense(512))
        model.add(Dense(128))
        model.add(Dense(10))
        model.add(Activation('softmax'))
        return model
    model = model()

    model.compile(loss='categorical_crossentropy',optimizer='adadelta',metrics=['accuracy'])
    # model.fit(trainX, trainY, batch_size=256, epochs=12,verbose=1, validation_data=(testX, testY))

    model =  keras.models.load_model('./Dense_4/model')
    # print(model.summary())
    # 模型测试
    score = model.evaluate(testX, testY, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

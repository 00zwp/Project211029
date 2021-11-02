from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import os
import numpy as np
from keras.utils import np_utils

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
tf.random.set_seed(666)
densemodel = keras.Sequential(
    [
        keras.Input(shape=(28, 28, 1)),
        layers.Flatten(),
        layers.Dense(1024),
        layers.Dense(512),
        layers.Dense(128),
        layers.Dense(10),
        layers.Activation('softmax'),
    ],
    name='densemodel',
)
tf.random.set_seed(666)
copymodel = keras.Sequential(
    [
        keras.Input(shape=(28, 28, 1)),
        layers.Flatten(),
        layers.Dense(1024),
        layers.Dense(512),
        layers.Dense(128),
        layers.Dense(10),
        layers.Activation('softmax'),
    ],
    name='copymodel',
)

optimizer = Adam(learning_rate=0.001)
loss_fn = keras.losses.categorical_crossentropy


def train_step(real_images, labels):
    with tf.GradientTape() as tape:
        predictions = densemodel(real_images)
        loss = loss_fn(labels, predictions)
    grads = tape.gradient(loss, densemodel.trainable_weights)
    predictions_sample = np.sum(np.equal(np.argmax(predictions.numpy(), 1), np.argmax(labels, 1)))
    optimizer.apply_gradients(zip(grads, densemodel.trainable_weights))
    optimizer.apply_gradients(zip(grads, copymodel.trainable_weights))
    return loss, predictions_sample

def evaluate_dense(dataset):
    predict = 0
    for _, test_data in enumerate(dataset):
        real_images, labels_in = test_data
        predictions_in = densemodel(real_images)
        predict += np.sum(np.equal(np.argmax(predictions_in.numpy(), 1), np.argmax(labels_in, 1)))
    print(predict/len(x_train))

def evaluate_copy(dataset):
    predict = 0
    for _, test_data in enumerate(dataset):

        real_images, labels_in = test_data
        predictions_in = copymodel(real_images)
        predict += np.sum(np.equal(np.argmax(predictions_in.numpy(), 1), np.argmax(labels_in, 1)))

    print(predict/len(x_train))


# Prepare the dataset. We use both the training & test MNIST digits.
batch_size = 64
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data(path='./dataset/mnist.npz')

x_train = np.reshape(x_train, (-1, 784))
x_test = np.reshape(x_test, (-1, 784))
y_train = np_utils.to_categorical(y_train, 10)  # to one_hot
y_test = np_utils.to_categorical(y_test, 10)  # to one_hot

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)

epochs = 10  # In practice you need at least 20 epochs to generate nice digits.
save_dir = "./"

for epoch in range(epochs):
    print("\nStart epoch", epoch)
    sum_predictions = 0
    for step, train_data in enumerate(train_dataset):
        realimages, labels = train_data
        # Train the discriminator & generator on one batch of real images.
        loss, predictions = train_step(realimages, labels)
        sum_predictions += predictions
        # Logging.
        # if step % 200 == 0:
            # Print metrics
            # print("discriminator loss at step %d: %.2f" % (step, np.average(loss.numpy())))
    print(sum_predictions/len(x_train))
    evaluate_dense(train_dataset)
    evaluate_copy(train_dataset)
    #如何获得evaluate
    #进一步考虑信息迁移
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import os
import numpy as np
from keras.utils import np_utils
import copy

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
if __name__ == '__main__':
    learning_rate = 0.00001
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

    def my_init(shape, dtype = None):
        weight = densemodel.weights[0]
        return weight

    def my_init1(shape, dtype = None):
        weight = densemodel.weights[0]
        return weight

    copymodel = keras.Sequential(
        [
            keras.Input(shape=(28, 28, 1)),
            layers.Flatten(),
            layers.Dense(1024, kernel_initializer=my_init),
            layers.Dense(1024),
            layers.Dense(512),
            layers.Dense(128),
            layers.Dense(10),
            layers.Activation('softmax'),
        ],
        name='copymodel',
    )
    print(copymodel)
    # 保证两个模型权重初始化相同，并且不能设置为 0
    quit(0)
    optimizer = Adam(learning_rate=learning_rate)
    loss_fn = keras.losses.categorical_crossentropy

    def train_step(real_images, labelsx):
        alpha = 0.99
        with tf.GradientTape() as tape:
            predictions = densemodel(real_images)
            loss = loss_fn(labelsx, predictions)
        grads = tape.gradient(loss, densemodel.trainable_weights)
        new_grad = []

        for i in range(len(new_grad)):
            if i < 2:
                new_grad.append((grads[i] * learning_rate - densemodel.trainable_weights[i] +
                                 model_for_pruning.trainable_weights[i]) / learning_rate)
            elif i < 6:
                if i < 4:
                    new_weight_train = tf.tile(densemodel.trainable_weights[i], [1, 2])
                    new_grad0_for_dense1024 = tf.tile(grads[i], [1, 2])
                    new_grad0_for_dense1024 = (new_grad0_for_dense1024 * learning_rate - alpha * new_grad0_for_dense1024
                                               * model_for_pruning.trainable_weights[i] * learning_rate) / (
                                                      new_weight_train * learning_rate - alpha *
                                                      new_grad0_for_dense1024 * learning_rate * learning_rate)
                    new_grad.append(new_grad0_for_dense1024)
                else:
                    new_grad.append((grads[i - 2] * learning_rate - densemodel.trainable_weights[i] +
                                     model_for_pruning.trainable_weights[i - 2]) / learning_rate)
            else:
                new_grad.append((grads[i - 2] * learning_rate - densemodel.trainable_weights[i] +
                                 model_for_pruning.trainable_weights[i - 2]) / learning_rate)
            grads[i] = 0.00001 * grads[i]
        optimizer.apply_gradients(zip(grads, densemodel.trainable_weights))
        optimizer.apply_gradients(zip(new_grad, model_for_pruning.trainable_weights))  # 优化器利用梯度训练copy模型
        del new_grad
        return loss


    def evaluate_dense(dataset):
        predict = 0
        sum = 0
        for _, test_data in enumerate(dataset):
            real_images, labels_in = test_data
            predictions_in = densemodel(real_images, training=False)
            predict += np.sum(np.equal(np.argmax(predictions_in.numpy(), 1), np.argmax(labels_in, 1)))
            sum += len(real_images)
        print(predict / sum)


    def evaluate_copy(dataset):
        predict = 0
        sum = 0
        for _, test_data in enumerate(dataset):
            real_images, labels_in = test_data
            predictions_in = model_for_pruning(real_images)
            predict += np.sum(np.equal(np.argmax(predictions_in.numpy(), 1), np.argmax(labels_in, 1)))
            sum += len(real_images)
        print(predict / sum)


    # Prepare the dataset. We use both the training & test MNIST digits.
    batch_size = 64
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data(path='./dataset/mnist.npz')

    x_train = np.reshape(x_train, (-1, 28, 28, 1))
    x_test = np.reshape(x_test, (-1, 28, 28, 1))
    y_train = np_utils.to_categorical(y_train, 10)  # to one_hot
    y_test = np_utils.to_categorical(y_test, 10)  # to one_hot

    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_dataset = test_dataset.shuffle(buffer_size=1024).batch(batch_size)
    epochs = 10  # In practice you need at least 20 epochs to generate nice digits.
    save_dir = "./"

    import tensorflow_model_optimization as tfmot

    prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude

    pruning_params = {
        'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=0.50,
                                                                 final_sparsity=0.80,
                                                                 begin_step=0,
                                                                 end_step=200000)
    }

    model_for_pruning = prune_low_magnitude(copymodel, **pruning_params)
    print(model_for_pruning.summary())
    for epoch in range(epochs):
        print("\nStart epoch", epoch)
        for step, train_data in enumerate(train_dataset):
            realimages, labels = train_data
            # Train the discriminator & generator on one batch of real images.
            loss = train_step(realimages, labels)

        evaluate_dense(train_dataset)
        evaluate_dense(test_dataset)
        evaluate_copy(train_dataset)
        evaluate_copy(test_dataset)

    # 进一步考虑信息迁移
    # 2021/11/2 暂时失败，可以加入模型剪枝，权重共享


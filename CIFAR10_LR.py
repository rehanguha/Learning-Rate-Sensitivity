from tensorflow import keras
from tensorflow.keras import datasets, models, layers
import matplotlib.pylab as plt
import tensorflow as tf
import numpy as np
import json
import pandas as pd
from tensorflow.keras.optimizers import SGD, RMSprop, Adam, Adagrad, Adadelta, Adamax, Nadam
import alert

mail = alert.mail(sender_email="rehan.logs@gmail.com", sender_password="rehanguhalogs")


lrs = np.linspace(0.001, 0.1, num = 100, endpoint =True).tolist()
epochs =[200] 
optimizers = [SGD, RMSprop, Adam, Adagrad, Adadelta, Adamax, Nadam]


(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1
x_train, x_test = x_train / 255.0, x_test / 255.0


def compile_optimizer(optimizer):
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10))

    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'])
    return model


def train_mnist(model, epoch):
    history = model.fit(x_train, y_train,
                    epochs=epoch,
                    verbose=True,
                    validation_data=(x_test, y_test))
    return history, model


mnist_df = pd.DataFrame(columns = ['optimizer', 'lr', 'epoch', 'accuracy', 'loss', 'test_accuracy', 'test_loss'])


for opt in optimizers:
    for lr in lrs:
        for epoch in epochs:
            model = compile_optimizer(opt(learning_rate=lr))
            history, model = train_mnist(model, epoch)
            train_loss, train_accuracy = model.evaluate(x_train, y_train, verbose=False)
            test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=False)
            mnist_df = mnist_df.append(
            {
              'optimizer': opt.__name__,
              'lr': lr,
              'epoch': epoch,
              'accuracy': train_accuracy,
              'loss': train_loss,
              'test_accuracy': test_accuracy,
              'test_loss': test_loss
            }, ignore_index= True)
    mnist_df.to_csv("output/CIFAR10_LR.csv", index=False)
    mail.send_email(receiver_email="rehan.guha@imaginea.com", subject="{name} Completed".format(name=opt.__name__,lr=lr), msg="Done.")

mnist_df.to_csv("output/CIFAR10_LR.csv", index=False)
    
mail.send_email(receiver_email="rehan.guha@imaginea.com", subject="All Completed".format(name=opt.__name__,lr=lr), msg="Saved.")




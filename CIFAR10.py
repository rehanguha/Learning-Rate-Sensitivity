import alert
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, models
import numpy as np
import json
from tensorflow.keras.optimizers import SGD, RMSprop, Adam, Adagrad, Adadelta, Adamax, Nadam
import logging

# create logger
logger = logging.getLogger('LOGFILE.log')
logger.setLevel(logging.DEBUG)

# create console handler and set level to debug
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

# create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# add formatter to ch
ch.setFormatter(formatter)

# add ch to logger
logger.addHandler(ch)


mail = alert.mail(sender_email="rehan.logs@gmail.com", sender_password="rehanguhalogs")

lrs = [0.1, 0.03, 0.01, 0.003, 0.001]
epochs = [50, 100, 150, 200, 250]
optimizers = [RMSprop, Adam, SGD, Adagrad, Adadelta, Adamax, Nadam]


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
    
    logger.info('Model Compiled')

    return model

def train_mnist(model, epoch):
    history = model.fit(x_train, y_train,
                    epochs=epoch,
                    verbose=True,
                    validation_data=(x_test, y_test))
    logger.info('Model Fit Complete')

    return history, model


(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1
x_train, x_test = x_train / 255.0, x_test / 255.0


logger.info('Dataset Load and Split Complete')

for epoch in epochs:
    for opt in optimizers:
        for lr in lrs:
            mnist_key = "{opt}_{lr}_{epoch}".format(opt=opt.__name__,lr=lr,epoch=epoch)
            logger.info("---")
            logger.info(mnist_key)
            logger.info("Started...")
            model = compile_optimizer(opt(learning_rate=lr))
            history, model = train_mnist(model, epoch)
            with open("output/CIFAR10/{name}_history.json".format(name=mnist_key), "w") as outputfile:
                json.dump(history.history, outputfile)
            logger.info("Completed.")
            logger.info("---")
    mail.send_email(receiver_email="rehan.guha@imaginea.com", subject="{name}_{epoch} Completed".format(name=opt.__name__,epoch=epoch), msg="All LR Completed.")
mail.send_email(receiver_email="rehan.guha@imaginea.com", subject="== Completed ==".format(name=opt.__name__,epoch=epoch), msg="Benchmark saved.")
    

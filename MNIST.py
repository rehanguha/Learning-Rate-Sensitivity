import alert
from tensorflow import keras
from tensorflow.keras import datasets
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
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=x_train[0].shape),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(
        optimizer=optimizer,
        loss=keras.losses.categorical_crossentropy,
        metrics=['accuracy'])
    
    logger.info('Model Compiled')

    return model

def train_mnist(model, epoch):
    history = model.fit(x_train, y_train,
                    batch_size=64,
                    epochs=epoch,
                    verbose=True,
                    shuffle=True,
                    validation_data=(x_val, y_val))
    logger.info('Model Fit Complete')

    return history, model


(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()

y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

x_train = keras.utils.normalize(x_train, axis=1)
x_test = keras.utils.normalize(x_test, axis=1)

# Reserve 10,000 samples for validation
x_val = x_train[-10000:]
y_val = y_train[-10000:]
x_train = x_train[:-10000]
y_train = y_train[:-10000]

logger.info('Dataset Load and Split Complete')

for opt in optimizers:
  for lr in lrs:
    for epoch in epochs:
        mnist_key = "{opt}_{lr}_{epoch}".format(opt=opt.__name__,lr=lr,epoch=epoch)
        logger.info("---")
        logger.info(mnist_key)
        logger.info("Started...")
        model = compile_optimizer(opt(learning_rate=lr))
        history, model = train_mnist(model, epoch)
        with open("output/MNIST/{name}_history.json".format(name=mnist_key), "w") as outputfile:
            json.dump(history.history, outputfile)
        logger.info("Completed.")
        logger.info("---")
    mail.send_email(receiver_email="rehan.guha@imaginea.com", subject="{name}_{lr} Completed".format(name=opt.__name__,lr=lr), msg="Benchmark saved.")


import flwr as fl
import tensorflow as tf
# Importing the necessary libraries, which we may or may not use. Its always good idea to import them befor (if you remember) else we can do it at any point of time no problem.
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout, Input, AveragePooling2D, Activation, Conv2D, MaxPooling2D, BatchNormalization, Concatenate
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras import regularizers, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import requests

from utils import randomprefix

import sys

if len(sys.argv) < 2:
    print("Please provide an integer argument.")
    sys.exit(1)

try:
    integer_arg = int(sys.argv[1])
    print(f"You entered the integer: {integer_arg}")

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    # model = tf.keras.applications.MobileNetV2(
    #     (32, 32, 3), classes=10, weights=None)
    model = Sequential()
    model.add(MaxPool2D(pool_size=(16, 16), input_shape=(32, 32, 3)))
    model.add(Flatten())
    model.add(Dense(10, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    model.summary()
    model.compile("adam", "sparse_categorical_crossentropy",
                  metrics=["accuracy"])

    model2 = Sequential()
    model2.add(MaxPool2D(pool_size=(16, 16), input_shape=(32, 32, 3)))
    model2.add(Flatten())
    model2.add(Dense(10, activation='softmax'))
    model2.compile(loss='categorical_crossentropy',
                   optimizer='adam', metrics=['accuracy'])
    model2.summary()
    model2.compile("adam", "sparse_categorical_crossentropy",
                   metrics=["accuracy"])

    url = f"http://167.71.139.232:900{integer_arg}/api/update-dataset/"

    class CifarClient(fl.client.NumPyClient):
        def get_parameters(self, config):
            return model.get_weights()

        def fit(self, parameters, config):
            print("Fit called", parameters, config)
            model.set_weights(parameters)
            model.fit(x_train, y_train, epochs=1,
                      batch_size=32, steps_per_epoch=3)
            weights = model.get_weights()
            smpc_weights = []
            for arr in [w.flatten().tolist() for w in weights]:
                smpc_weights.extend(arr)
            print("Weights client side", smpc_weights)
            data = {
                "type": "float",
                "data": smpc_weights
            }
            response = requests.post(
                url + "testKey" + randomprefix + str(config["round"]), json=data)
            if response.ok:
                print("Request was successful!")
                print(response.text)
            else:
                print(
                    f"Request failed with status code {response.status_code}.")
                print(response.text)
            return model2.get_weights(), len(x_train), {}

        def evaluate(self, parameters, config):
            model.set_weights(parameters)
            loss, accuracy = model.evaluate(x_test, y_test)
            return loss, len(x_test), {"accuracy": float(accuracy)}

    fl.client.start_numpy_client(
        server_address="[::]:8080", client=CifarClient())

except ValueError:
    print("The provided argument is not a valid integer.")

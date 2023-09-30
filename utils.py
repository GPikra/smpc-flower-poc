from typing import Tuple, Union, List
import numpy as np
from sklearn.linear_model import LogisticRegression
import openml
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# from tensorflow.keras.datasets import mnist

XY = Tuple[np.ndarray, np.ndarray]
Dataset = Tuple[XY, XY]
LogRegParams = Union[XY, Tuple[np.ndarray]]
XYList = List[XY]


def get_model_parameters(model):
    """Returns the paramters of a sklearn LogisticRegression model"""
    if model.fit_intercept:
        params = (model.coef_[0], model.intercept_)
    else:
        params = (model.coef_[0],)

    print("PARAMETERS", params[0], params[1])
    return params


def set_model_params(
    model: LogisticRegression, params: LogRegParams
) -> LogisticRegression:
    """Sets the parameters of a sklean LogisticRegression model"""
    model.coef_ = params[0]
    if model.fit_intercept:
        model.intercept_ = params[1]
    return model


def set_initial_params(model: LogisticRegression):
    """
    Sets initial parameters as zeros
    """
    n_classes = 2  # MNIST has 10 classes
    n_features = 4  # Number of features in dataset
    model.classes_ = np.array([i for i in range(2)])

    model.coef_ = np.zeros((n_classes, n_features))
    if model.fit_intercept:
        model.intercept_ = np.zeros((n_classes,))


def load_mnist() -> Dataset:
    """
    Loads the MNIST dataset using OpenML
    Dataset link: https://www.openml.org/d/554
    """
    X, y = load_iris(return_X_y=True)
    # digits = load_digits()

    # x_train, x_test, y_train, y_test = train_test_split(
    #     digits.data, digits.target, test_size=0.25, random_state=0)
    # # mnist_openml = openml.datasets.get_dataset(554)
    # # Xy, _, _, _ = mnist_openml.get_data(dataset_format="array")
    # X = Xy[:, :-1]  # the last column contains labels
    # y = Xy[:, -1]
    # # First 60000 samples consist of the train set
    x_train, y_train = X[:50], y[:50]
    x_test, y_test = X[50:], y[50:]
    return (x_train, y_train), (x_test, y_test)


def partition(X: np.ndarray, y: np.ndarray, num_partitions: int) -> XYList:
    """Split X and y into a number of partitions."""
    return list(
        zip(np.array_split(X, num_partitions),
            np.array_split(y, num_partitions))
    )


def partition(X: np.ndarray, y: np.ndarray, num_partitions: int) -> XYList:
    """Split X and y into a number of partitions."""
    return list(
        zip(np.array_split(X, num_partitions),
            np.array_split(y, num_partitions))
    )

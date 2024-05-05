from enum import Enum
from typing import Tuple
import numpy as np


class MemristorFault(Enum):
    IDEAL = 0
    DISCORDANT = 1
    STUCK = 2
    CONCORDANT = 3


def model_to_use_for_fault_classification():
    return 2


def fit_zero_intercept_lin_model(x: np.ndarray, y: np.ndarray) -> float:
    """
    :param x: x coordinates of data points (i.e., $\Delta R_i^\text{ideal}$)
    :param y: y coordinates of data points (i.e., \Delta R_i$)
    :return: theta 
    """

    return np.mean(x * y) / np.mean(x * x)


def bonus_fit_lin_model_with_intercept_using_pinv(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """
    :param x: x coordinates of data points (i.e., $\Delta R_i^\text{ideal}$)
    :param y: y coordinates of data points (i.e., \Delta R_i$)
    :return: theta_0, theta_1
    """
    from numpy.linalg import pinv

    x_train = np.column_stack((x, np.ones(x.shape)))
    theta = pinv(x_train) @ y
    return theta[1], theta[0]


def fit_lin_model_with_intercept(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """
    :param x: x coordinates of data points (i.e., $\Delta R_i^\text{ideal}$)
    :param y: y coordinates of data points (i.e., \Delta R_i$)
    :return: theta_0, theta_1 
    """
    theta_1 = (np.mean(x * y) - np.mean(y) * np.mean(x)) / (np.mean(x * x) - np.mean(x) * np.mean(x))
    theta_0 = np.mean(y) - theta_1 * np.mean(x)
    return theta_0, theta_1


def classify_memristor_fault_with_model1(theta: float) -> MemristorFault:
    """
    :param theta: the estimated parameter of the zero-intercept linear model
    :return: the type of fault
    """
    if np.isclose(1.0, theta):
        return MemristorFault.IDEAL
    elif np.abs(theta) < 0.1:
        return MemristorFault.STUCK
    elif theta < 0:
        return MemristorFault.DISCORDANT
    else:
        return MemristorFault.CONCORDANT


def classify_memristor_fault_with_model2(theta0: float, theta1: float) -> MemristorFault:
    """
    :param theta0: the intercept parameter of the linear model
    :param theta1: the slope parameter of the linear model
    :return: the type of fault
    """
    if np.isclose(1.0, theta1):
        return MemristorFault.IDEAL
    elif np.abs(theta1) < 0.1:
        return MemristorFault.STUCK
    elif theta1 < 0:
        return MemristorFault.DISCORDANT
    else:
        return MemristorFault.CONCORDANT

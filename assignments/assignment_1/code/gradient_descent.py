import numpy as np


def gradient_descent(f, df, x0, y0, learning_rate, lr_decay, num_iters):
    """
    Find a local minimum of the function f(x) using gradient descent:
    Until the number of iteration is reached, decrease the current x and y points by the
    respective partial derivative times the learning_rate.
    In each iteration, record the current function value in the list f_list.
    The function should return the minimizing argument (x, y) and f_list.

    :param f: Function to minimize
    :param df: Gradient of f i.e, function that computes gradients
    :param x0: initial x0 point
    :param y0: initial y0 point
    :param learning_rate: Learning rate
    :param lr_decay: Learning rate decay
    :param num_iters: number of iterations
    :return: x, y (solution), f_list (array of function values over iterations)
    """
    f_list = np.zeros(num_iters) # Array to store the function values over iterations
    x, y = x0, y0
    for i in range(num_iters):
        gradient_x, gradient_y = df(x, y)
        x -= learning_rate * gradient_x
        y -= learning_rate * gradient_y
        learning_rate *= lr_decay
        f_list[i] = f(x, y)

    return x, y, f_list


def ackley(x, y):
    """
    Ackley function at point (x, y)
    :param x: X coordinate
    :param y: Y coordinate
    :return: f(x, y) where f is the Ackley function
    """
    part_1 = -20 * np.exp(-0.2 * np.sqrt(0.5 * (x**2 + y**2)))
    part_2 = -np.exp(0.5 * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y)))
    return part_1 + part_2 + np.e + 20


def gradient_ackley(x, y):
    """
    Compute the gradient of the Ackley function at point (x, y)
    :param x: X coordinate
    :param y: Y coordinate
    :return: \nabla f(x, y) where f is the Ackley function
    """
    # Calculate components common to both partial derivatives
    common_factor_A = 2 * np.exp(-0.2 * np.sqrt(0.5 * (x**2 + y**2)))
    sqrt_xy = np.sqrt(0.5 * (x**2 + y**2))
    if sqrt_xy == 0:
        dx_A = 0
        dy_A = 0
    else:
        dx_A = common_factor_A * (x / sqrt_xy)
        dy_A = common_factor_A * (y / sqrt_xy)
    
    common_factor_B = np.pi * np.exp(0.5 * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y)))
    dx_B = common_factor_B * np.sin(2 * np.pi * x)
    dy_B = common_factor_B * np.sin(2 * np.pi * y)

    df_dx = dx_A + dx_B
    df_dy = dy_A + dy_B

    gradient = np.array([df_dx, df_dy])
    return gradient

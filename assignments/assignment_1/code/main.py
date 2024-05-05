import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from plot_utils import plot_model1, plot_model2, plot_logistic_regression, plot_datapoints, plot_function
from lin_reg_memristors import (model_to_use_for_fault_classification, fit_zero_intercept_lin_model,
                                fit_lin_model_with_intercept, bonus_fit_lin_model_with_intercept_using_pinv,
                                classify_memristor_fault_with_model1, classify_memristor_fault_with_model2)
from gradient_descent import ackley, gradient_ackley, gradient_descent
from logistic_regression import (create_design_matrix_dataset_1,
                                 create_design_matrix_dataset_2,
                                 create_design_matrix_dataset_3,
                                 logistic_regression_params_sklearn)


def task_1():
    print('---- Task 1.1 ----')

    # Load the data
    data = np.load('data/memristor_measurements.npy')
    print(data.shape)
    
    n_memristor = data.shape[0]

    ### --- Use Model 1 (zero-intercept lin. model, that is, fit the model using fit_zero_intercept_lin_model)    
    estimated_params_per_memristor_model1 = np.zeros(n_memristor)
    for i in range(n_memristor):
        x, y = data[i, :, 0], data[i, :, 1]
        theta = fit_zero_intercept_lin_model(x, y)
        estimated_params_per_memristor_model1[i] = theta

    # Visualize the data and the best fit for each memristor
    plot_model1(data, estimated_params_per_memristor_model1)

    print('\nModel 1 (zero-intercept linear model).')
    print(f'Estimated theta per memristor: {estimated_params_per_memristor_model1}')

    ### --- Use Model 2 (lin. model with intercept, that is, fit the model using fit_lin_model_with_intercept)    
    estimated_params_per_memristor_model2 = np.zeros((n_memristor, 2))
    for i in range(n_memristor):
        x, y = data[i, :, 0], data[i, :, 1]
        theta0, theta1 = bonus_fit_lin_model_with_intercept_using_pinv(x, y)
        estimated_params_per_memristor_model2[i, :] = [theta0, theta1]

    # Visualize the data and the best fit for each memristor
    plot_model2(data, estimated_params_per_memristor_model2)

    print('\nModel 2 (linear model with intercept).')
    print(f"Estimated params (theta_0, theta_1) per memristor: {estimated_params_per_memristor_model2}")

    fault_types = []
    model_to_use = model_to_use_for_fault_classification()
    for i in range(n_memristor):
        if model_to_use == 1:
            fault_type = classify_memristor_fault_with_model1(estimated_params_per_memristor_model1[i])
        elif model_to_use == 2:
            fault_type = classify_memristor_fault_with_model2(estimated_params_per_memristor_model2[i, 0],
                                                              estimated_params_per_memristor_model2[i, 1])
        else:
            raise ValueError('Please choose either Model 1 or Model 2 for the decision on memristor fault type.')

        fault_types.append(fault_type)

    print(f'\nClassifications (based on Model {model_to_use})')
    for i, fault_type in enumerate(fault_types):
        print(f'Memristor {i+1} is classified as {fault_type}.')


def task_2():
    print('\n---- Task 2 ----')

    for task in [1, 2, 3]:
        print(f'---- Logistic regression task {task} ----')
        if task == 1:
            X_data = np.load("data/X-1-data.npy")
            y = np.load("data/targets-dataset-1.npy")
            create_design_matrix = create_design_matrix_dataset_1
        elif task == 2:
            X_data = np.load("data/X-1-data.npy")
            y = np.load("data/targets-dataset-2.npy")
            create_design_matrix = create_design_matrix_dataset_2
        elif task == 3:
            X_data = np.load("data/X-2-data.npy")
            y = np.load("data/targets-dataset-3.npy")
            create_design_matrix = create_design_matrix_dataset_3
        else:
            raise ValueError('Task not found.')

        X = create_design_matrix(X_data)
        

        # Plot the datapoints (just for visual inspection)
        plot_datapoints(X, y, f'Targets - Task {task}')

        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
        print(f'Shapes of: X_train {X_train.shape}, X_test {X_test.shape}, y_train {y_train.shape}, y_test {y_test.shape}')

        # Train the classifier
        custom_params = logistic_regression_params_sklearn()
        clf = LogisticRegression(**custom_params)
        clf.fit(X_train, y_train)
        acc_train, acc_test = clf.score(X_train, y_train), clf.score(X_test, y_test)

        print(f'Train accuracy: {acc_train * 100:.2f}%. Test accuracy: {100 * acc_test:.2f}%.')
        
        yhat_train = clf.predict_proba(X_train)
        yhat_test = clf.predict_proba(X_test)
        loss_train = log_loss(y_train, yhat_train)
        loss_test = log_loss(y_test, yhat_test)
        print(f'Train loss: {loss_train}. Test loss: {loss_test}.')

        plot_logistic_regression(clf, create_design_matrix, X_train, f'(Dataset {task}) Train set predictions',
                                 figname=f'logreg_train{task}')
        plot_logistic_regression(clf, create_design_matrix, X_test,  f'(Dataset {task}) Test set predictions',
                                 figname=f'logreg_test{task}')

        classifier_weights, classifier_bias = clf.coef_, clf.intercept_
        print(f'Parameters: {classifier_weights}, {classifier_bias}')


def task_3():
    print('\n---- Task 3 ----')

    # Plot the Function, to see how it looks like
    plot_function(ackley)

    # TODO: Choose a random starting point using samples from a standard normal distribution
    x0 = np.random.standard_normal()
    y0 = np.random.standard_normal()
    print(f'{x0:.4f}, {y0:.4f}')

    iterations = 1000
    lr = 0.1
    lr_decay = 0.99
    x, y, f_list = gradient_descent(ackley, gradient_ackley, x0, y0, lr, lr_decay, iterations)

    # Print the point that is found after `max_iter` solution
    print(f'{x:.4f}, {y:.4f}')

    figure = plt.plot(range(iterations), f_list)
    plt.xlabel("Iterations")
    plt.ylabel("Cost function")
    plt.title("Learning process")
    plt.savefig("plots/ackley_learning.png")
    plt.show()

    print(f'Solution found: f({x:.4f}, {y:.4f})= {ackley(x,y):.4f}' )
    print(f'Global optimum: f(0, 0)= {ackley(0,0):.4f}')


def main():
    np.random.seed(33761)

    # task_1()
    # task_2()
    task_3()


if __name__ == '__main__':
    main()

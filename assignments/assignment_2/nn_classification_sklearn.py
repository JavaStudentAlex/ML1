from typing import Tuple
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV
from matplotlib import pyplot as plt
import warnings
# We will suppress ConvergenceWarnings in this task. In practice, you should take warnings more seriously.
warnings.filterwarnings("ignore")

SEARCH_PARAMS = {
    'hidden_layer_sizes': [2, 10, 100, 200, 500],
}

ADVANCED_SEARCH_PARAMS = {
    'hidden_layer_sizes': [100, 200],
    'alpha': [0.0, 0.1, 1.0],
    'solver': ['adam', 'lbfgs']
}

def reduce_dimension(X_train: np.ndarray, n_components: int) -> Tuple[np.ndarray, PCA]:
    """
    :param X_train: Training data to reduce the dimensionality. Shape: (n_samples, n_features)
    :param n_components: Number of principal components
    :return: Data with reduced dimensionality, which has shape (n_samples, n_components), and the PCA object
    """
    pca = PCA(n_components=n_components, random_state=42)
    X_train_pca = pca.fit_transform(X_train)
    return X_train_pca, pca


def train_nn(X_train: np.ndarray, y_train: np.ndarray) -> MLPClassifier:
    """
    Train MLPClassifier with different number of neurons in one hidden layer.

    :param X_train: PCA-projected features with shape (n_samples, n_components)
    :param y_train: Targets
    :return: The MLPClassifier you consider to be the best
    """

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
                                                      test_size=0.2, random_state=42)
    
    train_size = X_train.shape[0]
    val_size = X_val.shape[0]
    train_indices = np.arange(train_size)
    test_indices = np.arange(train_size, train_size + val_size)
    cv = [(train_indices, test_indices)]

    new_x_train = np.concatenate((X_train, X_val), axis=0)
    new_y_train = np.concatenate((y_train, y_val), axis=0)

    nn = MLPClassifier(
        random_state=1,
        max_iter=500,
        solver='adam'
    )
    grid = GridSearchCV(nn, SEARCH_PARAMS, cv=cv, scoring='accuracy', verbose=3, return_train_score=True)
    grid.fit(new_x_train, new_y_train)

    return grid.best_estimator_


def train_nn_with_regularization(X_train: np.ndarray, y_train: np.ndarray) -> MLPClassifier:
    """
    Train MLPClassifier using regularization.

    :param X_train: PCA-projected features with shape (n_samples, n_components)
    :param y_train: Targets
    :return: The MLPClassifier you consider to be the best
    """
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
                                                      test_size=0.2, random_state=42)
    
    train_size = X_train.shape[0]
    val_size = X_val.shape[0]
    train_indices = np.arange(train_size)
    test_indices = np.arange(train_size, train_size + val_size)
    cv = [(train_indices, test_indices)]

    new_x_train = np.concatenate((X_train, X_val), axis=0)
    new_y_train = np.concatenate((y_train, y_val), axis=0)

    nn = MLPClassifier(
        random_state=1,
        max_iter=500,
        solver='adam',
        early_stopping=True,
        alpha=0.1
    )
    grid = GridSearchCV(nn, SEARCH_PARAMS, cv=cv, scoring='accuracy', verbose=3, return_train_score=True)
    grid.fit(new_x_train, new_y_train)

    return grid.best_estimator_


def plot_training_loss_curve(nn: MLPClassifier) -> None:
    """
    Plot the training loss curve.

    :param nn: The trained MLPClassifier
    """
    plt.figure()
    plt.plot(nn.loss_curve_)
    plt.title('Training Loss Curve')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.show()


def show_confusion_matrix_and_classification_report(nn: MLPClassifier, X_test: np.ndarray, y_test: np.ndarray) -> None:
    """
    Plot confusion matrix and print classification report.

    :param nn: The trained MLPClassifier you want to evaluate
    :param X_test: Test features (PCA-projected)
    :param y_test: Test targets
    """
    y_pred = nn.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=nn.classes_)
    disp.plot()
    plt.show()

    print(classification_report(y_test, y_pred))


def perform_grid_search(X_train: np.ndarray, y_train: np.ndarray) -> MLPClassifier:
    """
    Perform GridSearch using GridSearchCV.

    :param X_train: PCA-projected features with shape (n_samples, n_components)
    :param y_train: Targets
    :return: The best estimator (MLPClassifier) found by GridSearchCV
    """

    nn = MLPClassifier(
        random_state=42,
        max_iter=100,
    )
    grid = GridSearchCV(nn, ADVANCED_SEARCH_PARAMS, cv=5, scoring='accuracy', verbose=1)
    grid.fit(X_train, y_train)

    return grid.best_estimator_
from sklearn.model_selection import train_test_split
from mlp_classifier_own import MLPClassifierOwn
import numpy as np

def train_nn_own(X_train: np.ndarray, y_train: np.ndarray) -> MLPClassifierOwn:
    """
    Train MLPClassifierOwn with PCA-projected features.

    :param X_train: PCA-projected features with shape (n_samples, n_components)
    :param y_train: Targets
    :return: The MLPClassifierOwn object
    """
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
                                                      test_size=0.2, random_state=42)

    classifier = MLPClassifierOwn(
        num_epochs=5, 
        alpha=1.0,
        random_state=42,
        hidden_layer_sizes=(16,)

    )
    classifier.fit(X_train, y_train)

    train_accuracy = classifier.score(X_train, y_train)
    val_accuracy = classifier.score(X_val, y_val)
    
    print(f"Train Accuracy: {train_accuracy * 100:.2f}%")
    print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")

    return classifier
import random
from typing import List
from autodiff.neural_net import MultiLayerPerceptron
from autodiff.scalar import Scalar
import numpy as np

class MLPClassifierOwn():
    def __init__(self, num_epochs=5, alpha=0.0, batch_size=32,
                 hidden_layer_sizes=(100,), random_state=0):
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.alpha = alpha
        self.hidden_layer_sizes = hidden_layer_sizes
        self.random_state = random_state
        self.num_classes = None
        self.model = None

    @staticmethod
    def softmax(z: List[Scalar]) -> List[Scalar]:
        """
        Returns the softmax of the given list of Scalars (as another list of Scalars).

        :param z: List of Scalar values
        """
        raise NotImplementedError('Task 2.4 not implemented.')
        return None

    @staticmethod
    def multiclass_cross_entropy_loss(y_true: int, probs: List[Scalar]) -> Scalar:
        """
        Returns the multi-class cross-entropy loss for a single sample (as a Scalar).

        :param y_true: True class index (0-based)
        :param probs: List of Scalar values, representing the predicted probabilities for each class
        """
        raise NotImplementedError('Task 2.4 not implemented.')
        return None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the model on the given data.

        :param X: Features
        :param y: Targets
        """
        self.num_classes = len(set(y))

        random_idxs = np.random.permutation(X.shape[0])
        num_batches = X.shape[0] // self.batch_size # We will just skip the last batch if it's not a full batch

        self.model = MultiLayerPerceptron(num_inputs=X.shape[1],
                                          num_hidden=list(self.hidden_layer_sizes),
                                          num_outputs=self.num_classes)

        for epoch_idx in range(self.num_epochs):
            learning_rate = 1.0 - 0.9 * epoch_idx / self.num_epochs
            for batch_idx in range(num_batches):
                idxs_in_batch = random_idxs[self.batch_size*batch_idx:self.batch_size * (batch_idx + 1)]
                X_batch, y_batch = X[idxs_in_batch], y[idxs_in_batch]

                losses = []
                for xi, yi in zip(X_batch, y_batch):
                    out = self.model([Scalar(x) for x in xi])
                    probs = self.softmax(out)
                    sample_loss = self.multiclass_cross_entropy_loss(yi, probs)

                    losses.append(sample_loss)

                self.model.zero_grad()
                loss = np.mean(losses)
                loss.backward()

                for p in self.model.parameters():
                    p.value -= learning_rate * p.grad
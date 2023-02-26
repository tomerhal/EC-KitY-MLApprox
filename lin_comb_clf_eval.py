from eckity.sklearn_compatible.classification_evaluator import ClassificationEvaluator, CLASSIFICATION_THRESHOLD
from eckity.genetic_encodings.ga.vector_individual import Vector

from sklearn.metrics import balanced_accuracy_score

import numpy as np

class LinCombClassificationfEvaluator(ClassificationEvaluator):
    def classify_individual(self, individual: Vector) -> np.ndarray:
        """
        Classify a given individual's predictions on the dataset.

        The individual returns a a float value, that must be mapped to a class.
        The mapping is done by calcualating the dot product between the individual's
        vector and the dataset. If the dot product is greater than the threshold,
        the individual is classified as 1, otherwise it is classified as 0.

        Parameters
        ----------
        individual : Individual
            Individual from the population.

        Returns
        -------
        ndarray
            Classification predictions of the individual.
        """
        vec = individual.get_vector()
        scores = np.dot(self.X, np.array(vec))
        return np.where(scores > CLASSIFICATION_THRESHOLD, 1, 0)
    
    def _evaluate_individual(self, individual):
        y_pred = self.classify_individual(individual)
        return balanced_accuracy_score(y_true=self.y, y_pred=y_pred)

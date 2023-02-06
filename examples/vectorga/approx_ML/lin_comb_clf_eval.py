from eckity.sklearn_compatible.classification_evaluator import ClassificationEvaluator, CLASSIFICATION_THRESHOLD

import numpy as np

class LinCombClassificationfEvaluator(ClassificationEvaluator):
    def classify_individual(self, individual):
        vec = individual.get_vector()
        scores = np.dot(self.X, np.array(vec))
        return np.where(scores > CLASSIFICATION_THRESHOLD, 1, 0)

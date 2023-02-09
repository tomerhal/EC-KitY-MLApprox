"""
My Teza.
"""

import numpy as np
from time import time

from time import time
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer

from eckity.algorithms.simple_evolution import SimpleEvolution
from eckity.breeders.simple_breeder import SimpleBreeder
from eckity.subpopulation import Subpopulation
from eckity.creators.ga_creators.float_vector_creator import GAFloatVectorCreator
from eckity.statistics.best_average_worst_statistics import BestAverageWorstStatistics

from eckity.genetic_operators.selections.tournament_selection import TournamentSelection
from eckity.genetic_operators.crossovers.vector_k_point_crossover import VectorKPointsCrossover
from eckity.genetic_operators.mutations.vector_random_mutation import FloatVectorUniformNPointMutation
from eckity.genetic_operators.mutations.vector_random_mutation import FloatVectorGaussNPointMutation

from eckity.sklearn_compatible.sk_classifier import SKClassifier
from lin_comb_clf_eval import LinCombClassificationfEvaluator

from pmlb import fetch_data

from clf_ml_approx_eval import ClfMLApproxPopulationEvaluator


def main():
    """
    Basic setup.
    """

    start_time = time()

    # load the breast cancer dataset
    X, y = fetch_data('chess',return_X_y=True, local_cache_dir='./')

    # Initialize SimpleEvolution instance
    algo = SimpleEvolution(
        Subpopulation(creators=GAFloatVectorCreator(length=X.shape[1], bounds=(-100, 100)),
                      population_size=500,
                      # user-defined fitness evaluation method
                      evaluator=LinCombClassificationfEvaluator(),
                      # maximization problem (fitness is sum of values), so higher fitness is better
                      higher_is_better=True,
                      elitism_rate=0.0,
                      # genetic operators sequence to be applied in each generation
                      operators_sequence=[
                        #   VectorKPointsCrossover(probability=0.7, k=2),
                        #   FloatVectorUniformNPointMutation(probability=0.1, n=5),
                        #   FloatVectorGaussNPointMutation(probability=0.1, n=5)
                      ],
                      selection_methods=[
                          # (selection method, selection probability) tuple
                          (TournamentSelection(tournament_size=4, higher_is_better=True), 1)
                      ]),
        breeder=SimpleBreeder(),
        population_evaluator=ClfMLApproxPopulationEvaluator(),
        max_workers=1,
        max_generation=100,
        statistics=BestAverageWorstStatistics()
    )
    # wrap the basic evolutionary algorithm with a sklearn-compatible classifier
    classifier = SKClassifier(algo)

    # split the dataset to train and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # fit the model (perform evolution process)
    classifier.fit(X_train, y_train)

    # check training set results
    print(f'\nbest pure fitness over training set: {algo.best_of_run_.get_pure_fitness()}')

    # check test set results by computing the accuracy score between the prediction result and the test set result
    test_score = accuracy_score(y_test, classifier.predict(X_test))
    print(f'test score: {test_score}')

    print(f'Total runtime: {time() - start_time} seconds.')


if __name__ == "__main__":
    main()

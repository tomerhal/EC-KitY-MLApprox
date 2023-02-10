"""
My Teza.
"""

import numpy as np
from time import process_time

from time import time
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from eckity.algorithms.simple_evolution import SimpleEvolution
from eckity.breeders.simple_breeder import SimpleBreeder
from eckity.subpopulation import Subpopulation
from eckity.creators.ga_creators.float_vector_creator import GAFloatVectorCreator
from eckity.statistics.best_average_worst_statistics import BestAverageWorstStatistics
from approx_statistics import ApproxStatistics

from eckity.genetic_operators.selections.tournament_selection import TournamentSelection
from eckity.genetic_operators.crossovers.vector_k_point_crossover import VectorKPointsCrossover
from eckity.genetic_operators.mutations.vector_random_mutation import FloatVectorUniformNPointMutation
from eckity.genetic_operators.mutations.vector_random_mutation import FloatVectorGaussNPointMutation

from eckity.sklearn_compatible.sk_classifier import SKClassifier
from lin_comb_clf_eval import LinCombClassificationfEvaluator

from pmlb import fetch_data

from approx_ml_pop_eval import ApproxMLPopulationEvaluator


def main():
    """
    Basic setup.
    """

    evoml_start_time = process_time()

    # load the magic dataset
    X, y = fetch_data('magic',return_X_y=True, local_cache_dir='./')
    # split the dataset to train and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train) # scaled data has mean 0 and variance 1 (only over training set)
    X_test = sc.transform(X_test) # use same scaler as one fitted to training data

    ind_eval = LinCombClassificationfEvaluator()

    evoml = SimpleEvolution(
        Subpopulation(creators=GAFloatVectorCreator(length=X.shape[1], bounds=(-1, 1)),
                      population_size=100,
                      # user-defined fitness evaluation method
                      evaluator=ind_eval,
                      # maximization problem (fitness is sum of values), so higher fitness is better
                      higher_is_better=True,
                      elitism_rate=0.0,
                      # genetic operators sequence to be applied in each generation
                      operators_sequence=[
                        VectorKPointsCrossover(probability=0.7, k=2),
                        FloatVectorGaussNPointMutation(probability=0.3, n=5),
                        FloatVectorUniformNPointMutation(probability=0.1, n=5)
                      ],
                      selection_methods=[
                          # (selection method, selection probability) tuple
                          (TournamentSelection(tournament_size=4, higher_is_better=True), 1)
                      ]),
        breeder=SimpleBreeder(),
        population_evaluator=ApproxMLPopulationEvaluator(population_sample_size=10, accumulate_population_data=True),
        max_workers=1,
        max_generation=100,
        statistics=ApproxStatistics(ind_eval)
    )
    # wrap the basic evolutionary algorithm with a sklearn-compatible classifier
    evoml_classifier = SKClassifier(evoml)

    # train the classifier
    evoml_classifier.fit(X_train, y_train)

    # calculate the accuracy of the classifier
    y_pred = evoml_classifier.predict(X_test)
    accuracy = balanced_accuracy_score(y_test, y_pred)
    print("Balanced Accuracy: ", accuracy)

    evoml_time = process_time() - evoml_start_time
    print('Total time:', evoml_time)


if __name__ == "__main__":
    main()

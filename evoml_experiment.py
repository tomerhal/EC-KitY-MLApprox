import numpy as np
import sys
from time import process_time

from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import SGDRegressor, LinearRegression, Ridge

from eckity.algorithms.simple_evolution import SimpleEvolution
from eckity.breeders.simple_breeder import SimpleBreeder
from eckity.subpopulation import Subpopulation
from eckity.creators.ga_creators.float_vector_creator import GAFloatVectorCreator

from approx_statistics import ApproxStatistics
from plot_statistics import PlotStatistics
from utils import *

from eckity.genetic_operators.selections.tournament_selection import TournamentSelection
from eckity.genetic_operators.crossovers.vector_k_point_crossover import VectorKPointsCrossover
from eckity.genetic_operators.mutations.vector_random_mutation import FloatVectorUniformNPointMutation
from eckity.genetic_operators.mutations.vector_random_mutation import FloatVectorGaussNPointMutation

from eckity.sklearn_compatible.sk_classifier import SKClassifier
from lin_comb_clf_eval import LinCombClassificationfEvaluator

from pmlb import fetch_data

from approx_ml_pop_eval import ApproxMLPopulationEvaluator
from plateau_switch_condition import PlateauSwitchCondition

got_to_plateau = False
def main():
    """
    Basic setup.
    """

    evoml_start_time = process_time()

    # parse system arguments
    if len(sys.argv) < 2:
        print("Usage: python evoml_experiment.py <dataset_name>")
        exit(1)

    dsname = sys.argv[1]
    model_type = Ridge
    model_params = {'alpha': 100}

    # load the dataset
    X, y = fetch_data(dsname, return_X_y=True, local_cache_dir='datasets')
    # split the dataset to train and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)  # scaled data has mean 0 and variance 1 (only over training set)
    X_test = sc.transform(X_test)  # use same scaler as one fitted to training data

    ind_eval = LinCombClassificationfEvaluator()
    plateau = PlateauSwitchCondition(gens=5,threshold=0.01, switch_once=True)

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
        population_evaluator=ApproxMLPopulationEvaluator(population_sample_size=100,
                                                         gen_sample_step=5,
                                                         accumulate_population_data=True,
                                                         cache_fitness=False,
                                                         model_type=model_type,
                                                         model_params=model_params,
                                                         should_approximate=(lambda eval: plateau.should_approximate(eval) and eval.approx_fitness_error < thresholds[dsname]),
                                                         switch_once=True,
                                                         gen_weight=linear_gen_weight),
        max_workers=1,
        max_generation=100,
        statistics=ApproxStatistics(ind_eval)#PlotStatistics(),
    )
    # wrap the basic evolutionary algorithm with a sklearn-compatible classifier
    evoml_classifier = SKClassifier(evoml)

    # train the classifier
    evoml_classifier.fit(X_train, y_train)

    print(
        f'Approximations: {evoml_classifier.algorithm.population_evaluator.approx_count / evoml_classifier.algorithm.max_generation}')

    # calculate the accuracy of the classifier
    y_pred = evoml_classifier.predict(X_test)
    accuracy = balanced_accuracy_score(y_test, y_pred)
    print("Balanced Accuracy on test set: ", accuracy)

    evoml_time = process_time() - evoml_start_time
    print('Total time:', evoml_time)

    plot_stats = evoml.statistics[0]
    plot_stats.plot_statistics(dsname, model_type, model_params)


if __name__ == "__main__":
    main()

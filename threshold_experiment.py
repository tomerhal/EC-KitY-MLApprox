import numpy as np
import sys
from time import process_time
import matplotlib.pyplot as plt

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
import utils

from eckity.genetic_operators.selections.tournament_selection import TournamentSelection
from eckity.genetic_operators.crossovers.vector_k_point_crossover import VectorKPointsCrossover
from eckity.genetic_operators.mutations.vector_random_mutation import FloatVectorUniformNPointMutation
from eckity.genetic_operators.mutations.vector_random_mutation import FloatVectorGaussNPointMutation

from eckity.sklearn_compatible.sk_classifier import SKClassifier
from lin_comb_clf_eval import LinCombClassificationfEvaluator

from pmlb import fetch_data

from approx_ml_pop_eval import ApproxMLPopulationEvaluator
from plateau_switch_condition import PlateauSwitchCondition

def main():
    evoml_start_time = process_time()

    # parse system arguments
    if len(sys.argv) < 2:
        print("Usage: python datasets_experiment.py <n_iter>")
        exit(1)

    n_iter = int(sys.argv[1])
    thresholds = range(20, 81, 10)

    model_type = Ridge
    model_params = {'alpha': 2}

    # for recording scores and params across all replicates

    alliters = dict.fromkeys(thresholds)
    for k in alliters:
        alliters[k] = {'test_scores': [], 'full_times': []}

    for i in range(n_iter):
        print(f'\n\nITER {i}\n')
        for threshold in thresholds:

            test_scores = []
            full_times = []
            
            for dsname in utils.BIG_DATASETS:
                start_time = process_time()

                # load the dataset
                X, y = fetch_data(dsname, return_X_y=True, local_cache_dir='../../../EC-KitY-MLApprox-Old/datasets')
                # split the dataset to train and test set
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

                sc = StandardScaler()
                X_train = sc.fit_transform(X_train)  # scaled data has mean 0 and variance 1 (only over training set)
                X_test = sc.transform(X_test)  # use same scaler as one fitted to training data


                evoml = SimpleEvolution(
                    Subpopulation(creators=GAFloatVectorCreator(length=X.shape[1], bounds=(-1, 1)),
                                population_size=100,
                                # user-defined fitness evaluation method
                                evaluator=LinCombClassificationfEvaluator(),
                                # maximization problem, so higher fitness is better
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
                    population_evaluator=ApproxMLPopulationEvaluator(population_sample_size=20,
                                                                    gen_sample_step=1,
                                                                    accumulate_population_data=True,
                                                                    model_type=model_type,
                                                                    model_params=model_params,
                                                                    ensemble=True,
                                                                    gen_weight=utils.square_gen_weight,
                                                                    should_approximate=lambda eval: eval.gen > threshold),
                    max_workers=1,
                    max_generation=100
                )
                # wrap the basic evolutionary algorithm with a sklearn-compatible classifier
                evoml_classifier = SKClassifier(evoml)

                # train the classifier
                evoml_classifier.fit(X_train, y_train)

                # calculate the accuracy of the classifier
                y_pred = evoml_classifier.predict(X_test)
                accuracy = balanced_accuracy_score(y_test, y_pred)
                test_scores.append(accuracy)

                dataset_time = process_time() - start_time
                full_times.append(dataset_time)

            alliters[threshold]['test_scores'].append(np.mean(test_scores))
            alliters[threshold]['full_times'].append(np.mean(full_times))

    evoml_time = process_time() - evoml_start_time
    print('Total time:', evoml_time)

    mean_test_scores = [np.mean(alliters[threshold]['test_scores']) for threshold in thresholds]
    mean_full_times = [np.mean(alliters[threshold]['full_times']) for threshold in thresholds]

    plt.title(f'All Datasets {model_type.__name__} {model_params}')
    plt.plot(mean_test_scores, label='mean test score')
    # plt.plot(mean_full_times, label='mean time')
    plt.xlabel('approximation percentage')
    # plt.xticks(np.arange(0.2, 0.8, 0.1))
    plt.legend()
    plt.show()




if __name__ == "__main__":
    main()

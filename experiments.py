import numpy as np
from pathlib import Path
import os

from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor, LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor

from eckity.algorithms.simple_evolution import SimpleEvolution
from eckity.breeders.simple_breeder import SimpleBreeder
from eckity.subpopulation import Subpopulation
from eckity.creators.ga_creators.float_vector_creator import GAFloatVectorCreator
from eckity.genetic_operators.selections.tournament_selection import TournamentSelection
from eckity.genetic_operators.crossovers.vector_k_point_crossover import VectorKPointsCrossover
from eckity.genetic_operators.mutations.vector_random_mutation import FloatVectorUniformNPointMutation
from eckity.genetic_operators.mutations.vector_random_mutation import FloatVectorGaussNPointMutation
from eckity.sklearn_compatible.sk_classifier import SKClassifier

from pmlb import fetch_data
from time import process_time
from operator import itemgetter
from argparse import ArgumentParser
from mlxtend.evaluate import permutation_test

from approx_ml_pop_eval import ApproxMLPopulationEvaluator
from lin_comb_clf_eval import LinCombClassificationfEvaluator
from plot_statistics import PlotStatistics
from timed_population_evaluator import TimedPopulationEvaluator
from utils import *

def scoring(y_true, y_pred):
    return balanced_accuracy_score(y_true, y_pred)

def create_evoml_clf(n_features, model_type, model_params, dsname) -> SKClassifier:
    evoml = SimpleEvolution(
            Subpopulation(creators=GAFloatVectorCreator(length=n_features, bounds=(-1, 1)),
                        population_size=100,
                        # user-defined fitness evaluation method
                        evaluator=LinCombClassificationfEvaluator(),
                        # maximization problem (fitness is balanced accuracy), so higher fitness is better
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
                                                             gen_sample_step=1,
                                                             accumulate_population_data=True,
                                                             cache_fitness=False,
                                                             model_type=model_type,
                                                             model_params=model_params,
                                                             should_approximate=lambda eval: eval.approx_fitness_error < thresholds[dsname]),
            max_workers=1,
            max_generation=100,
            statistics=PlotStatistics()
        )
    # wrap the basic evolutionary algorithm with a sklearn-compatible classifier
    evoml_classifier = SKClassifier(evoml)
    return evoml_classifier

def create_evo_clf(n_features) -> SKClassifier:
    evo = SimpleEvolution(
        Subpopulation(creators=GAFloatVectorCreator(length=n_features, bounds=(-1, 1)),
                      population_size=100,
                      # user-defined fitness evaluation method
                      evaluator=LinCombClassificationfEvaluator(),
                      # maximization problem (fitness is balanced accuracy), so higher fitness is better
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
        population_evaluator=TimedPopulationEvaluator(),
        max_workers=1,
        max_generation=100,
        statistics=PlotStatistics()
    )

    # wrap the basic evolutionary algorithm with a sklearn-compatible classifier
    evo_classifier = SKClassifier(evo)
    return evo_classifier

def fprint(fname, s):
    # if stdin.isatty(): print(s) # running interactively 
    with open(Path(fname),'a') as f: f.write(s)

def save_params(fname, dsname, n_replicates, n_samples, n_features, model, model_params):
    fprint(fname, f' dsname: {dsname}\n n_samples: {n_samples}\n n_features: {n_features:}\n n_replicates: {n_replicates}\n\
 model: {model.__name__}\n model params: {model_params}\n approx threshold: {thresholds[dsname]}\n\n')

def get_args():
    parser = ArgumentParser()
    parser.add_argument('-resdir', dest='resdir', type=str, action='store', help='directory where results are placed', default='results')
    parser.add_argument('-dsname', dest='dsname', type=str, action='store', help='dataset name')
    parser.add_argument('-nrep', dest='n_replicates', type=int, action='store', help='number of replicate runs')
    args = parser.parse_args()
    if None in [getattr(args, arg) for arg in vars(args)]:
        parser.print_help()
        exit()
    resdir, dsname, n_replicates = args.resdir+'/', args.dsname, args.n_replicates
    fname = resdir + dsname + '.txt'

    if os.path.exists(fname):
        os.remove(fname)

    return fname, dsname, n_replicates

def tostring(clf):
    """
    Returns a string representation of the classifier.
    """
    pop_eval = clf.algorithm.population_evaluator
    return 'EvoML' if isinstance(pop_eval, ApproxMLPopulationEvaluator) else 'Evo'


def main():
    """
    Basic setup.
    """
    start_time = process_time()
    fname, dsname, n_replicates = get_args()

    model_type = Ridge
    model_params = {'alpha': 300}

    # load the dataset
    X, y = fetch_data(dsname, return_X_y=True, local_cache_dir='datasets')
    n_samples, n_features = X.shape
    save_params(fname, dsname, n_replicates, n_samples, n_features, model_type, model_params)

    allreps = dict.fromkeys(['Evo', 'EvoML']) # for recording scores and params across all replicates
    for k in allreps: 
        allreps[k] = {'test_scores': [], 'full_times': [], 'eval_times': []}
    approximations = []

    for rep in range(1, n_replicates + 1):
        evo_clf, evoml_clf = create_evo_clf(n_features), create_evoml_clf(n_features, model_type, model_params, dsname)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train) # scaled data has mean 0 and variance 1 (only over training set)
        X_test = sc.transform(X_test) # use same scaler as one fitted to training data

        for clf in [evo_clf, evoml_clf]:
            clf_start_time = process_time()
            clf.fit(X_train, y_train)

            if clf == evoml_clf:
                approximations.append(clf.algorithm.population_evaluator.approx_count / clf.algorithm.generation_num)

            y_pred = clf.predict(X_test)
            test_score = scoring(y_test, y_pred)

            clf_name = tostring(clf)
            allreps[clf_name]['test_scores'].append(test_score)

            eval_time = clf.algorithm.population_evaluator.evaluation_time
            allreps[clf_name]['eval_times'].append(eval_time)

            clf_end_time = process_time() - clf_start_time
            allreps[clf_name]['full_times'].append(clf_end_time)

            fprint(fname, f'rep {rep}, {clf_name}, score: {round(test_score,3)}, eval_time: {eval_time}, total time: {clf_end_time}\n')
        
    medians = [[clf, np.median(allreps[clf]['test_scores'])] for clf in ['Evo', 'EvoML']]
    medians = sorted(medians, key=itemgetter(1), reverse=True)
        
    # 10,000-round permutation test to assess statistical significance of diff in test scores between 1st and 2nd places
    pval = permutation_test(allreps[medians[0][0]]['test_scores'], allreps[medians[1][0]]['test_scores'], method='approximate', num_rounds=10_000,\
                            func=lambda x, y: np.abs(np.median(x) - np.median(y)))
    
    pp = ''
    if medians[0][0] == 'EvoML' and pval<0.05: # ranked first, statistically significant
        pp = '!'
    elif medians[1][0] == 'EvoML' and pval>=0.05: # ranked second, statistically insignificant
        pp = '=='

    s_res = f'*>> {dsname}, '
    for i, m in enumerate(medians):
        s_res += f'#{i+1}: {m[0]} {round(m[1],4)}'
        if m[0]=='EvoML':
            s_res += pp
        s_res += ', '

    fprint(fname, s_res[:-2] + '\n')

    fprint(fname, f'*Approximations: {round(np.mean(approximations), 3)}\n')

    runtime = process_time() - start_time
    fprint(fname, f'*Runtime {runtime}\n')


if __name__ == "__main__":
    main()

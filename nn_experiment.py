import torch
import torchvision
import torchvision.transforms as transforms

import sys

import ssl

import numpy as np
from time import time

from net import Net

from sklearn.linear_model import Ridge

from eckity.algorithms.simple_evolution import SimpleEvolution
from eckity.breeders.simple_breeder import SimpleBreeder
from eckity.subpopulation import Subpopulation
from eckity.creators.ga_creators.int_vector_creator import GAIntVectorCreator

from approx_statistics import ApproxStatistics
from plot_statistics import PlotStatistics
import utils

from eckity.genetic_operators.selections.tournament_selection import TournamentSelection
from eckity.genetic_operators.crossovers.vector_k_point_crossover import VectorKPointsCrossover
from eckity.genetic_operators.mutations.vector_random_mutation import IntVectorNPointMutation

from eckity.sklearn_compatible.sk_classifier import SKClassifier
from nn_evaluator import NeuralNetworkEvaluator

from approx_ml_pop_eval import ApproxMLPopulationEvaluator
from plateau_switch_condition import PlateauSwitchCondition


def main():
    evoml_start_time = time()

    if len(sys.argv) < 2:
        print('Usage: python nn_experiment.py <gpu | cpu>')
        exit(1)

    if sys.argv[1] not in ['cpu', 'gpu']:
        raise ValueError('Invalid argument: ' + sys.argv[1])

    dsname = 'CIFAR-10'
    model_type = Ridge
    model_params = {'alpha': 2}

    # load the dataset
    transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = 5
    
    ssl._create_default_https_context = ssl._create_unverified_context

    train_samples = np.random.choice(50000, size=utils.CIFAR10_TRAIN_SAMPLES, replace=False)
    test_samples = np.random.choice(10000, size=utils.CIFAR10_TEST_SAMPLES, replace=False)

    trainset = torchvision.datasets.CIFAR10(root='./datasets', train=True,
                                            download=True, transform=transform)
    trainset = torch.utils.data.Subset(trainset, train_samples)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=False, num_workers=1)

    testset = torchvision.datasets.CIFAR10(root='./datasets', train=False,
                                        download=True, transform=transform)
    testset = torch.utils.data.Subset(testset, test_samples)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=1)

    n_epochs = 10
    ind_eval = NeuralNetworkEvaluator(trainset, batch_size, n_epochs)

    # evolution -> ML switch condition
    evo_plateau = PlateauSwitchCondition(gens=10,threshold=0.005, switch_once=False)
    # ML -> evolution switch condition
    evoml_plateau = PlateauSwitchCondition(gens=5,threshold=0.05, switch_once=False)

    def should_approximate(eval):
        if eval.is_approx:
            return evoml_plateau.should_approximate(eval) and eval.approx_fitness_error < utils.thresholds[dsname]
        else:
            return evo_plateau.should_approximate(eval) and eval.approx_fitness_error < utils.thresholds[dsname]

    evoml = SimpleEvolution(
        Subpopulation(creators=GAIntVectorCreator(length=7,
                                                  bounds=[
                                                        (6, 9),    # conv1_out
                                                        (3, 5),     # conv1_kernel_size
                                                        (2, 3),     # pooling_kernel_size
                                                        (13, 19),    # conv2_out
                                                        (3, 5),     # conv2_kernel_size
                                                        (100, 140),   # fc2_in
                                                        (60, 100)    # fc3_in
                                                        ]),
                      population_size=20,
                      # user-defined fitness evaluation method
                      evaluator=ind_eval,
                      # maximization problem, so higher fitness is better
                      higher_is_better=True,
                      elitism_rate=0.0,
                      # genetic operators sequence to be applied in each generation
                      operators_sequence=[
                          VectorKPointsCrossover(probability=0.7, k=2),
                          IntVectorNPointMutation(probability=0.3, n=5)
                      ],
                      selection_methods=[
                          # (selection method, selection probability) tuple
                          (TournamentSelection(tournament_size=4, higher_is_better=True), 1)
                      ]),
        breeder=SimpleBreeder(),
        population_evaluator=ApproxMLPopulationEvaluator(population_sample_size=0.2,
                                                         gen_sample_step=1,
                                                         model_type=model_type,
                                                         model_params=model_params,
                                                         ensemble=False,
                                                         gen_weight=utils.square_gen_weight,
                                                         should_approximate=should_approximate,
                                                         handle_duplicates='ignore',
                                                         n_folds=3,
                                                         eval_method=sys.argv[1]),
        max_generation=50,
        statistics=ApproxStatistics(ind_eval)#PlotStatistics(),
    )

    evoml.evolve()

    print(f'Approximations: {evoml.population_evaluator.approx_count / evoml.max_generation}')

    # Evaluate performance of the best individual on the test set
    best_vector = [int(x) for x in evoml.best_of_run_.vector]
    best_model = Net(*best_vector)
    optimizer, criterion = ind_eval.init_optimizer_lossfn(best_model)

    # Train the model on the entire train set, then evaluate it on test set
    test_score = ind_eval.train_net(best_model, criterion, optimizer, trainloader, testloader)
    print('Accuracy on test set:', test_score)
    

    evoml_time = time() - evoml_start_time
    print('Total time:', evoml_time)

    stats = evoml.statistics[0]
    stats.plot_statistics(dsname, model_type, model_params)

if __name__ == "__main__":
    main()

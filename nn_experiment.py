import torch
import torchvision
import torchvision.transforms as transforms

import ssl

import numpy as np
from time import process_time

from net import Net

from sklearn.linear_model import Ridge

from eckity.algorithms.simple_evolution import SimpleEvolution
from eckity.breeders.simple_breeder import SimpleBreeder
from eckity.subpopulation import Subpopulation
from eckity.creators.ga_creators.int_vector_creator import GAIntVectorCreator

from approx_statistics import ApproxStatistics
from plot_statistics import PlotStatistics
from utils import *

from eckity.genetic_operators.selections.tournament_selection import TournamentSelection
from eckity.genetic_operators.crossovers.vector_k_point_crossover import VectorKPointsCrossover
from eckity.genetic_operators.mutations.vector_random_mutation import IntVectorNPointMutation

from eckity.sklearn_compatible.sk_classifier import SKClassifier
from nn_evaluator import NeuralNetworkEvaluator

from approx_ml_pop_eval import ApproxMLPopulationEvaluator
from plateau_switch_condition import PlateauSwitchCondition


def main():
    evoml_start_time = process_time()

    dsname = 'CIFAR-10'
    model_type = Ridge
    model_params = {'alpha': 2}

    # load the dataset
    transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = 5
    
    ssl._create_default_https_context = ssl._create_unverified_context

    train_samples = np.random.choice(50000, size=CIFAR10_TRAIN_SAMPLES, replace=False)
    test_samples = np.random.choice(10000, size=CIFAR10_TEST_SAMPLES, replace=False)

    trainset = torchvision.datasets.CIFAR10(root='./datasets', train=True,
                                            download=True, transform=transform)
    trainset = torch.utils.data.Subset(trainset, train_samples)

    testset = torchvision.datasets.CIFAR10(root='./datasets', train=False,
                                        download=True, transform=transform)
    testset = torch.utils.data.Subset(testset, test_samples)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=1)

    ind_eval = NeuralNetworkEvaluator(trainset, batch_size, 1)

    evo_plateau = PlateauSwitchCondition(gens=10,threshold=0.005, switch_once=False)
    evoml_plateau = PlateauSwitchCondition(gens=5,threshold=0.05, switch_once=False)

    def should_approximate(eval):
        if eval.is_approx:
            return evoml_plateau.should_approximate(eval) and eval.approx_fitness_error < thresholds[dsname]
        else:
            return evo_plateau.should_approximate(eval) and eval.approx_fitness_error < thresholds[dsname]

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
                      population_size=6,
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
                                                         accumulate_population_data=True,
                                                         cache_fitness=False,
                                                         model_type=model_type,
                                                         model_params=model_params,
                                                         ensemble=False,
                                                         gen_weight=square_gen_weight,
                                                         should_approximate=should_approximate),
        max_workers=1,
        max_generation=2,
        statistics=ApproxStatistics(ind_eval)#PlotStatistics(),
    )

    evoml.evolve()

    print(f'Approximations: {evoml.population_evaluator.approx_count / evoml.max_generation}')

    # calculate the accuracy of the classifier
    best_model = Net(*evoml.best_of_run_.vector)
    print('Accuracy on test set:', ind_eval.val_epoch(best_model, testloader))
    

    evoml_time = process_time() - evoml_start_time
    print('Total time:', evoml_time)

    plot_stats = evoml.statistics[0]
    plot_stats.plot_statistics(dsname, model_type, model_params)

if __name__ == "__main__":
    main()

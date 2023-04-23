import numpy as np
from time import process_time

from sklearn.linear_model import Ridge
from blackjack_vector_k_point_crossover import BlackjackVectorKPointsCrossover
from blackjack_vector_n_point_mutation import BlackjackFloatVectorGaussNPointMutation, BlackjackFloatVectorUniformNPointMutation

from eckity.algorithms.simple_evolution import SimpleEvolution
from eckity.breeders.simple_breeder import SimpleBreeder
from eckity.subpopulation import Subpopulation
from eckity.creators.ga_creators.float_vector_creator import GAFloatVectorCreator

from approx_ml_pop_eval import ApproxMLPopulationEvaluator
from plateau_switch_condition import PlateauSwitchCondition
from approx_statistics import ApproxStatistics
import utils

from eckity.genetic_operators.selections.tournament_selection import TournamentSelection
from eckity.genetic_operators.crossovers.vector_k_point_crossover import VectorKPointsCrossover
from eckity.genetic_operators.mutations.vector_random_mutation import FloatVectorUniformNPointMutation
from eckity.genetic_operators.mutations.vector_random_mutation import FloatVectorGaussNPointMutation


from blackjack_evaluator import BlackjackEvaluator
from blackjack_utils import STATE_ACTION_SPACE_SHAPE

def main():
    """
    Basic setup.
    """

    evoml_start_time = process_time()

    model_type = Ridge
    model_params = {'alpha': 2}

    ind_eval = BlackjackEvaluator(initial_epsilon=0.0, epsilon_decay=0.0, final_epsilon=0.0, n_episodes=100_000)

    evo_plateau = PlateauSwitchCondition(gens=10,threshold=0.005, switch_once=False)
    evoml_plateau = PlateauSwitchCondition(gens=5,threshold=0.05, switch_once=False)

    def should_approximate(eval):
        if eval.is_approx:
            return evoml_plateau.should_approximate(eval) and eval.approx_fitness_error < 0.1
        else:
            return evo_plateau.should_approximate(eval) and eval.approx_fitness_error < 0.1


    evoml = SimpleEvolution(
        Subpopulation(creators=GAFloatVectorCreator(length=np.prod(STATE_ACTION_SPACE_SHAPE), bounds=(0, 1)),
                      population_size=10,
                      # user-defined fitness evaluation method
                      evaluator=ind_eval,
                      # maximization problem, so higher fitness is better
                      higher_is_better=True,
                      elitism_rate=0.0,
                      # genetic operators sequence to be applied in each generation
                      operators_sequence=[
                          BlackjackVectorKPointsCrossover(probability=0.7, k=2),
                          BlackjackFloatVectorGaussNPointMutation(probability=0.3, n=5),
                          BlackjackFloatVectorUniformNPointMutation(probability=0.1, n=5)
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
                                                         use_gpu=False),
        max_workers=1,
        max_generation=10,
        statistics=ApproxStatistics(ind_eval)#PlotStatistics(),
    )
    evoml.evolve()

    print(f'Approximations: {evoml.population_evaluator.approx_count / evoml.max_generation}')

    evoml_time = process_time() - evoml_start_time
    print('Total time:', evoml_time)

    stats = evoml.statistics[0]
    stats.plot_statistics('Blackjack', model_type, model_params)

if __name__ == "__main__":
    main()

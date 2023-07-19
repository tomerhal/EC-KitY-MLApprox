import numpy as np
import sys

from sklearn.linear_model import Ridge, Lasso

from eckity.algorithms.simple_evolution import SimpleEvolution
from eckity.breeders.simple_breeder import SimpleBreeder
from eckity.subpopulation import Subpopulation

from approx_ml_pop_eval import ApproxMLPopulationEvaluator

from plateau_switch_condition import PlateauSwitchCondition
from cos_sim_switch_cond import CosSimSwitchCondition

from plot_statistics import PlotStatistics
import utils

from eckity.genetic_operators.selections.tournament_selection \
    import TournamentSelection
from eckity.genetic_operators.crossovers.vector_k_point_crossover \
    import VectorKPointsCrossover

from blackjack_creator import BlackjackVectorCreator
from blackjack_mutation import BlackjackMutation
from blackjack_evaluator import BlackjackEvaluator


def main():
    """
    Basic setup.
    """
    if len(sys.argv) != 7:
        print('Usage: python3 sample_size_experiment.py <job_id> <problem> \
              <model> <switch_method> <handle_duplicates> <sample_rate>')
        exit(1)

    job_id = sys.argv[1]
    problem = sys.argv[2]
    model = sys.argv[3]
    switch_method = sys.argv[4]
    handle_duplicates = sys.argv[5]
    sample_rate = float(sys.argv[6])

    if problem == 'blackjack':
        length = np.prod(utils.BLACKJACK_STATE_ACTION_SPACE_SHAPE)
        og_vector = [int(x) for x in np.array(utils.GYM_BLACKJACK_MATRIX).flatten()]
        creator = BlackjackVectorCreator(length=length,
                                         bounds=(0, 1),
                                         noisy=True,
                                         original_vector=og_vector,
                                         noise=0.1)
        ind_eval = BlackjackEvaluator(n_episodes=100_000, use_rewards=True)
        mutation = BlackjackMutation(probability=0.3, n=length//10)
    elif problem == 'frozenlake':
        raise NotImplementedError('FrozenLake not implemented yet')
    else:
        raise ValueError('Invalid problem ' + problem)
    
    model_type = Ridge if model == 'ridge' else Lasso
    model_params = {'alpha': 0.3, 'max_iter': 3000}

    if switch_method == 'cos_sim':
        cos_sim = CosSimSwitchCondition(threshold=0.9, switch_once=False)

        def should_approximate(eval):
            return cos_sim.should_approximate(eval)

    elif switch_method == 'plateau':
        evo_plateau = PlateauSwitchCondition(gens=10, threshold=500,
                                             switch_once=False)
        evoml_plateau = PlateauSwitchCondition(gens=5, threshold=1000,
                                               switch_once=False)

        def should_approximate(eval):
            if eval.is_approx:
                return evoml_plateau.should_approximate(eval)
            else:
                return evo_plateau.should_approximate(eval)

    elif switch_method == 'error':
        def should_approximate(eval):
            return eval.approx_fitness_error < 270

    elif switch_method == 'dataset':
        def should_approximate(eval):
            return eval.df.shape[0] >= 1000
    else:
        raise ValueError('Invalid switch method ' + switch_method)

    evoml = SimpleEvolution(
        Subpopulation(creators=creator,
                      population_size=100,
                      evaluator=ind_eval,
                      higher_is_better=True,
                      elitism_rate=0.0,
                      operators_sequence=[
                          VectorKPointsCrossover(probability=0.7, k=2),
                          mutation,
                      ],
                      selection_methods=[
                          # (selection method, selection probability) tuple
                          (TournamentSelection(tournament_size=4, higher_is_better=True), 1)
                      ]),
        breeder=SimpleBreeder(),
        population_evaluator=ApproxMLPopulationEvaluator(population_sample_size=sample_rate,
                                                         gen_sample_step=1,
                                                         model_type=model_type,
                                                         model_params=model_params,
                                                         gen_weight=utils.square_gen_weight,
                                                         should_approximate=should_approximate,
                                                         handle_duplicates=handle_duplicates,
                                                         job_id=job_id),
        executor='process',
        max_workers=10,
        max_generation=15,
        statistics=PlotStatistics()
    )
    pop_eval = evoml.population_evaluator
    evoml.register('evolution_finished', pop_eval.print_best_of_run)
    evoml.evolve()

    statistics = evoml.statistics[0]
    statistics.plot_statistics()

    print('dataset samples:', pop_eval.df.shape[0])
    pop_eval.export_dataset(utils.DATASET_PATH)


if __name__ == "__main__":
    main()

import numpy as np
import sys

from sklearn.linear_model import Ridge, Lasso

from eckity.algorithms.simple_evolution import SimpleEvolution
from eckity.breeders.simple_breeder import SimpleBreeder
from eckity.subpopulation import Subpopulation
from eckity.genetic_operators.selections.tournament_selection \
    import TournamentSelection
from eckity.genetic_operators.crossovers.vector_k_point_crossover \
    import VectorKPointsCrossover
from eckity.genetic_operators.mutations.vector_random_mutation \
    import BitStringVectorNFlipMutation, IntVectorNPointMutation

from approx_ml_pop_eval import ApproxMLPopulationEvaluator

from plateau_switch_condition import PlateauSwitchCondition
from cos_sim_switch_cond import CosSimSwitchCondition

from plot_statistics import PlotStatistics
import utils
from novelty_search import NoveltySearchCreator

from blackjack_creator import BlackjackVectorCreator
from blackjack_evaluator import BlackjackEvaluator

from frozen_lake_evaluator import FrozenLakeEvaluator
from frozen_lake_creator import FrozenLakeVectorCreator


def main():
    """
    Basic setup.
    """
    if len(sys.argv) < 6:
        print('Usage: python3 evoml_experiment.py <job_id> <problem> \
              <model> <alpha> <switch_method> \
              <sample_rate>')
        exit(1)

    job_id = sys.argv[1]
    problem = sys.argv[2]
    model = sys.argv[3]
    switch_method = sys.argv[4]
    sample_rate = float(sys.argv[5])

    novelty = 'novelty' in sys.argv

    sample_strategy = 'cosine' if switch_method == 'cosine' else 'random'
    handle_duplicates = 'ignore'

    if problem == 'blackjack':
        length = np.prod(utils.BLACKJACK_STATE_ACTION_SPACE_SHAPE)
        ind_eval = BlackjackEvaluator(n_episodes=100_000)
        creator = BlackjackVectorCreator(length=length, bounds=(0, 1))
        mutation = BitStringVectorNFlipMutation(probability=0.3, n=length//10)

    elif problem == 'frozenlake':
        length = utils.FROZEN_LAKE_STATES
        ind_eval = FrozenLakeEvaluator(total_episodes=2000)
        creator = FrozenLakeVectorCreator(length=length, bounds=(0, 3))
        mutation = IntVectorNPointMutation(probability=0.3, n=length//10)
            
    else:
        raise ValueError('Invalid problem ' + problem)
    
    operators_sequence = [
        VectorKPointsCrossover(probability=0.7, k=2),
        mutation,
    ]

    if novelty:
        novelty_creator = NoveltySearchCreator(
            operators_sequence=operators_sequence,
            length=creator.length,
            bounds=creator.bounds,
            vector_type=creator.type,
            fitness_type=creator.fitness_type,
            k=20,
            max_archive_size=500
        )
        del creator
        creator = novelty_creator

    model_type = Ridge if model == 'ridge' \
        else Lasso
    
    model_params = {'alpha': 0.3, 'max_iter': 3000} if model == 'ridge' \
        else {'alpha': 0.65, 'max_iter': 1000} if model == 'lasso' \
        else {}

    if switch_method == 'cosine':
        cos_sim = CosSimSwitchCondition(threshold=0.9
                                        if problem == 'blackjack'
                                        else 0.95,
                                        switch_once=False)

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

    elif switch_method == 'true':
        def should_approximate(eval):
            return True
    else:
        raise ValueError('Invalid switch method ' + switch_method)

    evoml = SimpleEvolution(
        Subpopulation(creators=creator,
                      population_size=100,
                      evaluator=ind_eval,
                      higher_is_better=True,
                      elitism_rate=0.0,
                      operators_sequence=operators_sequence,
                      selection_methods=[
                          # (selection method, selection probability) tuple
                          (TournamentSelection(tournament_size=4,
                                               higher_is_better=True), 1)
                      ]),
        breeder=SimpleBreeder(),
        population_evaluator=ApproxMLPopulationEvaluator(
                population_sample_size=sample_rate,
                gen_sample_step=1,
                sample_strategy=sample_strategy,
                model_type=model_type,
                model_params=model_params,
                gen_weight=utils.sqrt_gen_weight,
                should_approximate=should_approximate,
                handle_duplicates=handle_duplicates,
                job_id=job_id),
        executor='process',
        max_workers=10,
        max_generation=200,
        statistics=PlotStatistics(),
    )
    pop_eval = evoml.population_evaluator
    evoml.register('evolution_finished', pop_eval.print_best_of_run)
    evoml.evolve()

    try:
        statistics = evoml.statistics[0]
        statistics.plot_statistics()
    except Exception as e:
        print('Failed to print statistics. Error:', e)

    try:
        print('dataset samples:', pop_eval.df.shape[0])
        pop_eval.export_dataset(utils.DATASET_PATH)
    except Exception as e:
        print('Failed to export dataset. Error:', e)

    print('fitness computations:', pop_eval.fitness_count)
    print('approximations:', pop_eval.approx_count)


if __name__ == "__main__":
    main()

import pytest
import pandas as pd
import numpy as np
import utils

from concurrent.futures import ProcessPoolExecutor

from eckity.fitness.simple_fitness import SimpleFitness
from eckity.evaluators.simple_individual_evaluator import SimpleIndividualEvaluator
from eckity.algorithms.simple_evolution import SimpleEvolution
from eckity.breeders.simple_breeder import SimpleBreeder
from eckity.subpopulation import Subpopulation
from eckity.genetic_operators.crossovers.vector_k_point_crossover import VectorKPointsCrossover
from eckity.genetic_operators.mutations.vector_random_mutation import BitStringVectorNFlipMutation
from eckity.genetic_operators.selections.tournament_selection import TournamentSelection
from blackjack_creator import BlackjackVectorCreator

from blackjack_individual import BlackjackIndividual
from blackjack_evaluator import BlackjackEvaluator
from approx_ml_pop_eval import ApproxMLPopulationEvaluator


class MyEvaluator(SimpleIndividualEvaluator):
    def evaluate_individual(self, individual):
        return np.random.random()


class Tests:
    def test_cos_sim(self):
        ind_vectors = [list(range(i, i+2)) for i in range(3)]
        df_vectors = [list(range(i, i+2)) for i in range(3, 6)]
        df = pd.DataFrame(df_vectors)
        df['fitness'] = [100, 200, 300]
        df['gen'] = [1, 1, 1]

        genotypes = df.iloc[:, :-2]
        scores = [
            genotypes.apply(lambda row: utils.cosine_similarity(ind_vector, row), axis=1, raw=True).max()
            for ind_vector in ind_vectors
        ]

        expected = np.array([0.8, 11 * np.sqrt(5) / 25, 18 * np.sqrt(13) / 65])
        assert all(np.isfinite(scores))
        assert all(np.abs(np.array(scores) - expected) < 1e-10)

    def test_fitness_clone(self):
        ind_len = np.prod(utils.BLACKJACK_STATE_ACTION_SPACE_SHAPE)
        ind = BlackjackIndividual(SimpleFitness(), length=ind_len, bounds=(0, 1), rewards=None)
        ind.fitness.set_fitness(100.0)
        assert ind.fitness.get_pure_fitness() == 100.0
        assert ind.fitness._is_evaluated

        ind_copy = ind.clone()
        assert ind_copy.fitness.get_pure_fitness() == 100.0
        assert ind_copy.fitness._is_evaluated

        assert ind_copy is not ind
        assert ind_copy.fitness is not ind.fitness
        assert ind_copy.fitness.get_pure_fitness() == ind.fitness.get_pure_fitness()

    def test_dataset_size(self):
        test_eval = BlackjackEvaluator(n_episodes=10)
        ind_len = np.prod(utils.BLACKJACK_STATE_ACTION_SPACE_SHAPE)
        test_evo = SimpleEvolution(
            Subpopulation(creators=BlackjackVectorCreator(length=ind_len),
                          population_size=2,
                          evaluator=test_eval,
                          higher_is_better=True,
                          elitism_rate=0.0,
                          operators_sequence=[
                              VectorKPointsCrossover(probability=0.7, k=2),
                              BitStringVectorNFlipMutation(probability=0.3, n=3),
                          ],
                          selection_methods=[
                              # (selection method, selection probability) tuple
                              (TournamentSelection(tournament_size=4, higher_is_better=True), 1)
                          ]),
            breeder=SimpleBreeder(),
            population_evaluator=ApproxMLPopulationEvaluator(test_eval,
                                                             eval_method='process'),
            max_generation=3)
        test_evo.evolve()
        pop_eval = test_evo.population_evaluator
        assert pop_eval.df.shape[0] <= 8

    def test_rewards_executor(self):
        executor = ProcessPoolExecutor(max_workers=1)
        ind_len = np.prod(utils.BLACKJACK_STATE_ACTION_SPACE_SHAPE)
        individuals = [BlackjackIndividual(SimpleFitness(), length=ind_len, bounds=(0, 1), rewards=None)]
        ind_eval = BlackjackEvaluator(n_episodes=10)
        fitness_scores = executor.map(ind_eval.evaluate_individual, individuals)
        ids2rewards = ind_eval.ids2rewards
        ind_id = individuals[0].id
        assert ind_id in ids2rewards
        assert ids2rewards[ind_id] == fitness_scores[0]

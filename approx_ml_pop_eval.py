import random
import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error
from sklearn.base import RegressorMixin
from sklearn.model_selection import KFold

from eckity.evaluators.simple_individual_evaluator import SimpleIndividualEvaluator
from eckity.sklearn_compatible.classification_evaluator import ClassificationEvaluator
from sklearn.linear_model import SGDRegressor
from overrides import overrides

from eckity.evaluators.individual_evaluator import IndividualEvaluator
from eckity.evaluators.population_evaluator import PopulationEvaluator
from eckity.fitness.fitness import Fitness
from eckity.individual import Individual
from eckity.population import Population

from typing import List


class ApproxMLPopulationEvaluator(PopulationEvaluator):
    """
    Fitness approximation population evaluator

    Parameters
    ----------
    approx_condition : callable, optional
        whether the fitness should be approximated in the current generation, by default None
    population_sample_size : int or float, optional
        number (or percentage) of individuals to sample and compute their fitness when approximating , by default 10
    gen_sample_step : int, optional
        how many generations should pass between samples, by default 1
    scoring : callable, optional
        evaluation metric for the model, by default mean_absolute_error
    accumulate_population_data : bool, optional
        whether to accumulate the population data for the model, by default False
    """
    def __init__(self,
                approx_condition: callable = None,
                population_sample_size=10,
                gen_sample_step=1,
                scoring=mean_absolute_error):
        super().__init__()
        self.approx_fitness_error = float('inf')
        self.population_sample_size = population_sample_size
        self.gen_sample_step = gen_sample_step
        self.scoring = scoring

        self.model = None
        self.gen = 0

        if approx_condition is None:
            self.should_approximate = lambda: self.approx_fitness_error < 0.1

    @overrides
    def _evaluate(self, population: Population) -> Individual:
        """
        Updates the fitness score of the given individuals, then returns the best individual

        Parameters
        ----------
        population:
            the population of the evolutionary experiment

        Returns
        -------
        Individual
            the individual with the best fitness out of the given individuals
        """
        super()._evaluate(population)
        for sub_population in population.sub_populations:
            if self.should_approximate():
                print('=== Approximating fitness ===')
                # Approximate fitness scores of the whole population
                preds = self.predict(sub_population.individuals)
                for i, ind in enumerate(sub_population.individuals):
                    ind.fitness.set_fitness(preds[i])
                
                if self.gen > 0 and self.gen % self.gen_sample_step == 0:
                    # Sample a subset of the population and compute their fitness
                    sample_size = self.population_sample_size if isinstance(self.population_sample_size, int) else int(len(sub_population.individuals) * self.population_sample_size)
                    sample_inds = random.sample(sub_population.individuals, sample_size)
                    for ind in sample_inds:
                        ind.set_fitness_not_evaluated()
                    fitnesses = self._evaluate_individuals(sample_inds, sub_population.evaluator)

                    # update the model's performance
                    self._update_model_error(sample_inds, fitnesses)
            else:
                print('=== Computing fitness ===')
                # Compute fitness scores of the whole population
                fitnesses = self._evaluate_individuals(sub_population.individuals, sub_population.evaluator)
                self.fit(sub_population.individuals, fitnesses)
            
            print('model mean absolute error:', self.approx_fitness_error)
        
        self.gen += 1

        # only one subpopulation in simple case
        individuals = population.sub_populations[0].individuals

        best_ind: Individual = population.sub_populations[0].individuals[0]
        best_fitness: Fitness = best_ind.fitness

        for ind in individuals[1:]:
            if ind.fitness.better_than(ind, best_fitness, best_ind):
                best_ind = ind
                best_fitness = ind.fitness

        return best_ind
    
    def _update_model_error(self, individuals: List[Individual], fitnesses):
        # update the model's performance
        err = self.eval(individuals, fitnesses)
        self.approx_fitness_error = err
    
    def _evaluate_individuals(self, individuals: List[Individual], evaluator: IndividualEvaluator) -> List[float]:
        """
        Evaluate the fitness scores of a given individuals list

        Parameters
        ----------
        individuals : List[Individual]
            list of individuals

        Returns
        -------
        List[float]
            list of fitness scores, with respect to the order of the individuals
        """
        # Evaluate the fitness and train the model incrementally
        eval_futures = [
            self.executor.submit(evaluator.evaluate, ind, individuals)
            for ind in individuals
        ]

        # wait for all fitness values to be evaluated before continuing
        for future in eval_futures:
            future.result()

        fitnesses = [ind.get_pure_fitness() for ind in individuals]
        return fitnesses

    def fit(self, individuals: List[Individual], fitnesses: List[float]) -> RegressorMixin:
        """
        Fit the Machine Learning model incrementally.

        The model is trained to estimate the fitness score of an individual given its representation.

        Parameters
        ----------
        individuals : List[Individual]
            List of individuals in the sub-population
        fitnesses : List[float]
            Fitness scores of the individuals, repectively
        """
        self.model = SGDRegressor(max_iter=1000, tol=1e-3)
        ind_vectors = [ind.get_vector() for ind in individuals]
        X, y = np.array(ind_vectors), np.array(fitnesses)

        scores = []
        kf = KFold(n_splits=5, shuffle=True)
        for train_index, test_index in kf.split(X):
            self.model.fit(X[train_index], y[train_index])            
            scores.append(self.scoring(y[test_index], self.model.predict(X[test_index])))
        self.approx_fitness_error = np.mean(scores)

        self.model.fit(ind_vectors, fitnesses)

    def predict(self, individuals: List[Individual]):
        """
        Perform fitness approximation of a given list of individuals.

        Parameters
        ----------
        individuals :   list of individuals
            Individuals in the sub-population

        Returns
        -------
        ndarray of shape (n_samples,)
           Predicted target values per element in X.
        """
        ind_vectors = [ind.get_vector() for ind in individuals]
        return self.model.predict(ind_vectors)

    def eval(self, individuals: List[Individual], fitnesses: List[float]) -> float:
        """
        Evaluate the Machine Learning model.

        This model predicts the fitness scores of individuals given their representation (genome).

        Parameters
        ----------
        individuals : List[Individual]
            List of individuals in the sub-population
        fitnesses : List[float]
            Fitness scores of the individuals, repectively

        Returns
        -------
        float
            accuracy score of the model
        """
        y_pred = self.predict(individuals)
        err = self.scoring(y_true=fitnesses, y_pred=y_pred)
        return err

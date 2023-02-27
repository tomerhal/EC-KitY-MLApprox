import random
import numpy as np
import pandas as pd
from time import process_time

from sklearn.metrics import mean_absolute_error
from sklearn.base import RegressorMixin
from sklearn.model_selection import KFold, cross_val_score

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
                should_approximate: callable = None,
                population_sample_size=10,
                gen_sample_step=1,
                scoring=mean_absolute_error,
                model_type=SGDRegressor,
                model_params=None,
                accumulate_population_data=False,
                cache_fitness=False):
        super().__init__()
        self.approx_fitness_error = float('inf')
        self.population_sample_size = population_sample_size
        self.gen_sample_step = gen_sample_step
        self.scoring = scoring

        if cache_fitness and not accumulate_population_data:
            raise ValueError('cache_fitness can only be enabled when accumulate_population_data is enabled')

        self.accumulate_population_data = accumulate_population_data
        self.cache_fitness = cache_fitness

        if model_params is None:
            model_params = {}
        self.model_params = model_params
        self.model_type = model_type
        self.model = None
        self.gen = 0
        self.evaluation_time = 0

        self.approx_count = 0
        self.gen_population = []
        self.best_in_gen = None

        if should_approximate is None:
            should_approximate = lambda eval: eval.approx_fitness_error < 0.1
        self.should_approximate = should_approximate

        if accumulate_population_data:
            self.df = None

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
        eval_start_time = process_time()
        super()._evaluate(population)
        self.gen_population = population
        if self.gen > 1:
            should_approximate = self.should_approximate(self)
        else:
            should_approximate = False
        for sub_population in population.sub_populations:
            if should_approximate:

                self.approx_count += 1
                # Approximate fitness scores of the whole population
                preds = self.predict(sub_population.individuals)
                for i, ind in enumerate(sub_population.individuals):
                    ind.fitness.set_fitness(preds[i])
                
                if self.gen > 0 and self.gen % self.gen_sample_step == 0:
                    # Sample a subset of the population and compute their fitness
                    sample_size = self.population_sample_size if isinstance(self.population_sample_size, int) else int(len(sub_population.individuals) * self.population_sample_size)
                    sample_inds = random.sample(sub_population.individuals, sample_size)
                    fitnesses = self._evaluate_individuals(sample_inds, sub_population.evaluator)

                    # update population dataframe with sampled individuals
                    if self.accumulate_population_data:
                        vecs = [ind.get_vector() for ind in sample_inds]
                        self._update_dataframe(vecs, fitnesses)

                    # update model's performance
                    self._update_model_error(sample_inds, fitnesses)
                
            else:
                # Compute fitness scores of the whole population
                fitnesses = self._evaluate_individuals(sub_population.individuals, sub_population.evaluator)
                for i, ind in enumerate(sub_population.individuals):
                    ind.fitness.set_fitness(fitnesses[i])
                self.fit(sub_population.individuals, fitnesses)

        self.gen += 1

        # only one subpopulation in simple case
        individuals = population.sub_populations[0].individuals

        best_ind: Individual = population.sub_populations[0].individuals[0]
        best_fitness: Fitness = best_ind.fitness

        for ind in individuals[1:]:
            if ind.fitness.better_than(ind, best_fitness, best_ind):
                best_ind = ind
                best_fitness = ind.fitness

        eval_end_time = process_time()
        self.evaluation_time += eval_end_time - eval_start_time
        self.best_in_gen = best_ind
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
        if self.gen > 0 and self.cache_fitness:
            # search for individuals in the first n-1 columns of the dataframe
            # the last column is the fitness
            df = self.df
            
            for ind in individuals:
                if ind.vector in df.values[:, :-1]:
                    ind.fitness.set_fitness(df[df.values[:, :-1] == ind.vector].values[0][-1])
            
        # Evaluate the fitness of the individuals that have not been evaluated yet
        # (if caching is not enabled, this will be all individuals)
        eval_futures = [
            self.executor.submit(evaluator._evaluate_individual, ind)
            for ind in individuals
        ]

        fitnesses = [future.result() for future in eval_futures]
        return fitnesses
    
    def _update_dataframe(self, ind_vectors: List[List], fitnesses: List[float]):
        if self.df is None:
                self.df = pd.DataFrame(np.array(ind_vectors))
                self.df['fitness'] = np.array(fitnesses)
        else:
            df = pd.DataFrame(np.array(ind_vectors))
            df['fitness'] = np.array(fitnesses)

            self.df = pd.concat([self.df, df], ignore_index=True, copy=False)
            n_features= self.df.shape[1] - 1

            # if the same individual is evaluated multiple times, keep the last evaluation
            self.df.drop_duplicates(subset=range(n_features), keep='last', inplace=True)

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
        self.model = self.model_type(**self.model_params)
        ind_vectors = [ind.get_vector() for ind in individuals]
        
        if self.accumulate_population_data:
            self._update_dataframe(ind_vectors, fitnesses)
            X, y = self.df.iloc[:, :-1].to_numpy(), self.df.iloc[:, -1].to_numpy()
        else:
            X, y = np.array(ind_vectors), np.array(fitnesses)

        # Too slow
        # self.approx_fitness_error = cross_val_score(self.model, X, y, cv=KFold(n_splits=5, shuffle=True)).mean()
        
        scores = []
        kf = KFold(n_splits=5, shuffle=True)
        for train_index, test_index in kf.split(X):
            self.model.fit(X[train_index], y[train_index])            
            scores.append(self.scoring(y[test_index], self.model.predict(X[test_index])))
        self.approx_fitness_error = np.mean(scores)

        self.model.fit(X, y)

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

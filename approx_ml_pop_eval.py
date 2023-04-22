import random
import numpy as np
import pandas as pd
from time import process_time
import subprocess
import os

from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold

from sklearn.linear_model import SGDRegressor
from overrides import overrides

from eckity.evaluators.simple_individual_evaluator import SimpleIndividualEvaluator
from eckity.evaluators.population_evaluator import PopulationEvaluator
from eckity.fitness.fitness import Fitness
from eckity.individual import Individual
from eckity.population import Population

from typing import List
import utils


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
    """

    def __init__(self,
                 should_approximate: callable = None,
                 population_sample_size=10,
                 gen_sample_step=1,
                 scoring=mean_absolute_error,
                 model_type=SGDRegressor,
                 model_params=None,
                 gen_weight=lambda gen: gen + 1,
                 ensemble=False,
                 n_folds=5,
                 handle_duplicates='ignore',
                 sort_encoding=False,
                 use_gpu=True):
        super().__init__()
        self.approx_fitness_error = float('inf')
        self.population_sample_size = population_sample_size
        self.gen_sample_step = gen_sample_step
        self.scoring = scoring

        self.ensemble = ensemble

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
        self.is_approx = False

        self.df = None

        if ensemble:
            self.models = dict()

        self.model_error_history = dict()
        self.gen_weight = gen_weight
        self.n_folds = n_folds

        self.handle_duplicates = handle_duplicates

        self.use_gpu = use_gpu

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
        if self.gen > 0:
            self.is_approx = self.should_approximate(self)
        else:
            self.is_approx = False

        for sub_population in population.sub_populations:
            if self.is_approx:
                self.approx_count += 1

                # TODO REMOVE THIS - FOR DEBUGGING ONLY
                with open('approx_count.txt', 'a') as f:
                    f.write(f'{self.approx_count}')

                # Approximate fitness scores of the whole population
                preds = self.predict(sub_population.individuals)
                for i, ind in enumerate(sub_population.individuals):
                    ind.fitness.set_fitness(preds[i])

                if self.gen > 0 and self.gen % self.gen_sample_step == 0:
                    # Sample a subset of the population and compute their fitness
                    sample_size = self.population_sample_size if isinstance(self.population_sample_size, int) else int(
                        len(sub_population.individuals) * self.population_sample_size)

                    if sample_size > 0:
                        sample_inds = random.sample(sub_population.individuals, sample_size)
                        fitnesses = self._evaluate_individuals(sample_inds, sub_population.evaluator)

                        # update population dataframe with sampled individuals
                        vecs = [ind.get_vector() for ind in sample_inds]
                        self._update_dataframe(vecs, fitnesses)

                        # train the model with the sampled individuals
                        self.fit(sample_inds, fitnesses)

            else:
                # Compute fitness scores of the whole population
                fitnesses = self._evaluate_individuals(sub_population.individuals, sub_population.evaluator)
                for ind, fitness_score in zip(sub_population.individuals, fitnesses):
                    ind.fitness.set_fitness(fitness_score)
                self.fit(sub_population.individuals, fitnesses)

        # only one subpopulation in simple case
        individuals = population.sub_populations[0].individuals

        best_ind = individuals[0]

        gen_df = self.df[self.df['gen'] == self.gen]
        if gen_df.empty:
            return None
        else:
            best_fitness_idx = gen_df['fitness'].idxmax()
            best_fitness = gen_df.loc[best_fitness_idx]['fitness']
            best_vector = gen_df.loc[best_fitness_idx][:-2].to_list()
            best_ind = best_ind.clone()
            best_ind.set_vector(best_vector)
            best_ind.fitness.set_fitness(best_fitness)

        self.best_in_gen = self._get_best_individual(individuals)

        self.gen += 1

        eval_end_time = process_time()
        self.evaluation_time += eval_end_time - eval_start_time

        return best_ind

    def _get_best_individual(self, individuals: List[Individual]) -> Individual:
        best_ind: Individual = individuals[0]
        best_fitness: Fitness = best_ind.fitness

        for ind in individuals[1:]:
            if ind.fitness.better_than(ind, best_fitness, best_ind):
                best_ind = ind
                best_fitness = ind.fitness
        return best_ind

    def _update_model_error(self, individuals: List[Individual], fitnesses):
        # update the model's performance
        err = self.eval(individuals, fitnesses)
        self.model_error_history[self.gen] = err
        gens, errors = self.model_error_history.keys(), self.model_error_history.values()
        self.approx_fitness_error = np.average(list(errors), weights=[self.gen_weight(gen) for gen in gens])

    def _evaluate_individuals(self, individuals: List[Individual], evaluator: SimpleIndividualEvaluator) -> List[float]:
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

        device = 'gpu' if self.use_gpu else 'cpu'

        # Evaluate the fitness of the individuals that have not been evaluated yet
        subprocesses = []
        num_inds_main_proc = int(len(individuals) / 20)

        # create a job script for the rest of the individuals and submit them
        for i, ind in enumerate(individuals[num_inds_main_proc:]):
            # create a job script for the individual
            vector = ind.get_vector()
            sbatch_str = utils.generate_sbatch_str(self.gen, i, vector, self.use_gpu)
            with open(f'jobs/{device}_{self.gen}_{i}.sh', 'w') as f:
                f.write(sbatch_str)

            # Submit the job
            cmdline = ['sbatch', f'jobs/{device}_{self.gen}_{i}.sh']
            subprocesses.append(subprocess.Popen(cmdline))

        # evaluate the first 5% individuals in the main process
        fitness_scores = [evaluator.evaluate_individual(ind) for ind in individuals[:num_inds_main_proc]]

        # Wait for all jobs to finish and extract fitness scores
        for i, process in enumerate(subprocesses):
            try:
                process.wait(timeout=360)

                # extract job id and fitness from job .out file
                with open(f'jobs/{device}_job_{self.gen}_{i}.out', 'r') as f:
                    job_id = f.readline().strip().split('=')[-1]
                    fitness = float(f.read())

                fitness_scores.append(fitness)
            except (subprocess.TimeoutExpired, FileNotFoundError) as e:
                print(f'Job {device}_{self.gen}_{i} timed out or failed. error: {e}')
                subprocess.Popen(['scancel', job_id])
                fitness_scores.append(0.0)
                continue

        # remove job scripts and output files
        for i in range(len(subprocesses)):
            try:
                os.remove(f'jobs/{device}_{self.gen}_{i}.sh')
                os.remove(f'jobs/{device}_job_{self.gen}_{i}.out')
            except FileNotFoundError as e:
                print(f'File not found in job {device}_{self.gen}_{i}:', e)

        return fitness_scores

    def _update_dataframe(self, ind_vectors: List[List], fitnesses: List[float]):
        df = pd.DataFrame(np.array(ind_vectors))
        df['fitness'] = np.array(fitnesses)
        df['gen'] = self.gen

        if self.df is None:
            self.df = df

        else:
            self.df = pd.concat([self.df, df], ignore_index=True, copy=False)
            n_features = self.df.shape[1] - 2

            # Handle rows with duplicate individuals:
            if self.handle_duplicates in ['last', 'first']:
                # If the same individual is evaluated multiple times, keep the first/last evaluation.
                self.df.drop_duplicates(subset=range(n_features), keep=self.handle_duplicates, inplace=True)

    def fit(self, individuals: List[Individual], fitnesses: List[float]) -> None:
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

        # Add new model to the ensemble (if ensemble is enabled)
        if self.ensemble:
            self.models[self.gen] = self.model

        ind_vectors = [ind.get_vector() for ind in individuals]

        # Vector of generation number of each individual in the dataframe (used for sample weights)
        w = None

        self._update_dataframe(ind_vectors, fitnesses)
        X, y = self.df.iloc[:, :-2].to_numpy(), self.df['fitness'].to_numpy()
        w = self.gen_weight(self.df['gen'].to_numpy())

        # Too slow
        # self.approx_fitness_error = cross_val_score(self.model, X, y, cv=KFold(n_splits=5, shuffle=True)).mean()

        scores = []
        kf = KFold(n_splits=self.n_folds, shuffle=True)
        for train_index, test_index in kf.split(X):
            sample_weight = w[train_index]
            self.model.fit(X[train_index], y[train_index], sample_weight)
            scores.append(self.scoring(y[test_index], self.model.predict(X[test_index])))
        self.approx_fitness_error = np.mean(scores)

        # Now fit the model on the whole training set
        self.model.fit(X, y, sample_weight=w)

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

        if self.ensemble:
            weights = [self.gen_weight(gen) for gen in self.models]
            preds = [model.predict(ind_vectors) for model in self.models.values()]
            preds = np.average(preds, weights=weights, axis=0)

        else:
            preds = self.model.predict(ind_vectors)

        # enforce the model's prediction to be between 0 and 1
        return [1.0 if pred > 1 else 0.0 if pred < 0 else pred for pred in preds]

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

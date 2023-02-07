from sklearn.metrics import accuracy_score

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


class ClfMLApproxPopulationEvaluator(PopulationEvaluator):
    """
    Compute the fitness of an individual in classification problems.
    """
    def __init__(self, ind_eval):
        super().__init__()
        self.ind_eval = ind_eval
        self.model = SGDRegressor(max_iter=1000, tol=1e-3)
        self.approx_fitness_accuracy = 0
        self.should_approximate = False

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
            if self.should_approximate:
                # Approximate fitness scores of the whole population
                preds = self.predict(sub_population.individuals)
                for i, ind in enumerate(sub_population.individuals):
                    ind.fitness.set_fitness(preds[i])
            else:
                # Evaluate the fitness and train the model incrementally
                sp_eval: IndividualEvaluator = sub_population.evaluator
                eval_futures = [
                    self.executor.submit(sp_eval.evaluate, ind, sub_population.individuals)
                    for ind in sub_population.individuals
                ]

                # wait for all fitness values to be evaluated before continuing
                for future in eval_futures:
                    future.result()

                fitnesses = [ind.get_pure_fitness() for ind in sub_population.individuals]

                # bug: calling predict before first partial_fit
                accuracy = self.eval(sub_population.individuals, fitnesses)

                self.approx_fitness_accuracy = accuracy
                self.partial_fit(sub_population.individuals, fitnesses)
                    
        # only one subpopulation in simple case
        individuals = population.sub_populations[0].individuals

        best_ind: Individual = population.sub_populations[0].individuals[0]
        best_fitness: Fitness = best_ind.fitness

        for ind in individuals[1:]:
            if ind.fitness.better_than(ind, best_fitness, best_ind):
                best_ind = ind
                best_fitness = ind.fitness

        return best_ind

    def partial_fit(self, individuals: List[Individual], fitnesses: List[float]) -> None:
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
        ind_vectors = [ind.get_vector() for ind in individuals]
        self.model.partial_fit(ind_vectors, fitnesses)

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
        accuracy = accuracy_score(y_true=fitnesses, y_pred=y_pred)
        print('model accuracy:', accuracy)
        return accuracy

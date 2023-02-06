from sklearn.metrics import accuracy_score

from eckity.evaluators.simple_individual_evaluator import SimpleIndividualEvaluator
from eckity.sklearn_compatible.classification_evaluator import ClassificationEvaluator
from sklearn.linear_model import SGDClassifier
from overrides import overrides

from eckity.evaluators.individual_evaluator import IndividualEvaluator
from eckity.evaluators.population_evaluator import PopulationEvaluator
from eckity.fitness.fitness import Fitness
from eckity.individual import Individual


class ClassificationMLapproxPopulationEvaluator(PopulationEvaluator):
    """
    Class to compute the fitness of an individual in classification problems.
    """

    def __init__(self, X=None, y=None):
        super().__init__()
        self.simple_fitness = ClassificationEvaluator()
        self.ml_model = SGDClassifier(max_iter=1000, tol=1e-3)
        self.approx_fitness_score = 0
        self.X = X
        self.y = y

    @overrides
    def _evaluate(self, population):
        """
        Updates the fitness score of the given individuals, then returns the best individual

        Parameters
        ----------
        population:
            the population of the evolutionary experiment

        Returns
        -------
        individual
            the individual with the best fitness out of the given individuals
        """
        super()._evaluate(population)
        for sub_population in population.sub_populations:
            if True:
                self.approx_fitness_score = self.eval(population.sub_populations[0].individuals)
                self.train_ml_model(population.sub_populations[0].individuals)

                sp_eval: IndividualEvaluator = sub_population.evaluator
                eval_futures = [
                    self.executor.submit(sp_eval.evaluate, ind, sub_population.individuals)
                    for ind in sub_population.individuals
                ]

                # wait for all fitness values to be evaluated before returning from this method
                for future in eval_futures:
                    future.result()
            else:
                for ind in sub_population.individuals:
                    ind.fitness.set_fitness(self.predict([ind])[0])
        # only one subpopulation in simple case
        individuals = population.sub_populations[0].individuals

        best_ind: Individual = population.sub_populations[0].individuals[0]
        best_fitness: Fitness = best_ind.fitness

        for ind in individuals[1:]:
            if ind.fitness.better_than(ind, best_fitness, best_ind):
                best_ind = ind
                best_fitness = ind.fitness

        return best_ind

    def train_ml_model(self, population):
        finesses = [self.simple_fitness._evaluate_individual(ind) for ind in population]
        ind_vectors = [ind.get_vector() for ind in population]
        self.ml_model.partial_fit(ind_vectors, finesses)

    def predict(self, ind):
        #ind_vectors = [ind.get_vector() for ind in population]
        ind_vectors = [ind]
        return self.ml_model.predict(ind_vectors)[0]

    def eval(self, population):
        print(population[0])
        finesses = [self.simple_fitness._evaluate_individual(ind) for ind in population]
        ind_vectors = [ind.get_vector() for ind in population]
        return accuracy_score(y_true=finesses, y_pred=self.predict(ind_vectors))

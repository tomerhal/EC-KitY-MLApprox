from eckity.termination_checkers.termination_checker import TerminationChecker


class PlatoTerminationChecker(TerminationChecker):
    def __init__(self, gens=5, threshold=0.01, higher_is_better=False):
        super().__init__()
        self.fitness_history = list()
        self.gens = gens
        self.threshold = threshold
        self.higher_is_better = higher_is_better

    def should_terminate(self, population, best_individual, gen_number):
        """
        Determines if the currently best fitness is close enough to the target fitness.
        If so, recommends the algorithm to terminate early.

        Parameters
        ----------
        population: Population
            The evolutionary experiment population of individuals.

        best_individual: Individual
            The individual that has the best fitness of the current generation.

        gen_number: int
            Current generation number.

        Returns
        -------
        bool
            True if the algorithm should terminate early, False otherwise.
        """

        # Check if the best fitness has changed
        curr_fitness = best_individual.get_pure_fitness()

        # Don't terminate during the first gens generations
        if self.gens > gen_number or len(self.fitness_history) < self.gens:
            self.fitness_history.append(curr_fitness)
            return False

        del self.fitness_history[0]
        self.fitness_history.append(curr_fitness)
        return (max(self.fitness_history) - min(self.fitness_history)) <= self.threshold

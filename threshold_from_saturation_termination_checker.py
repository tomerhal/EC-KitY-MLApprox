from eckity.termination_checkers.termination_checker import TerminationChecker


class ThresholdFromSaturationTerminationChecker(TerminationChecker):
    """
    Concrete Termination Checker that checks the distance from best existing fitness value to target fitness value.

    Parameters
    ----------
    optimal: float, default=0.
        Target fitness value.
        This termination checker checks if the currently best fitness is "close enough" to the optimal value.

    threshold: float, default=0.
        How close should the current best fitness be to the target fitness.

    higher_is_better: bool, default=False
        Determines if higher fitness values are better.
    """

    def __init__(self, saturation_time=0., threshold=0., higher_is_better=False):
        super().__init__()
        self.saturation_time_history = list(range(0, int(saturation_time*(threshold + 1)), int(threshold + 1)))
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
        self.saturation_time_history.pop()
        self.saturation_time_history.insert(0,best_individual.get_pure_fitness())
        return (max(self.saturation_time_history) - min(self.saturation_time_history)) <= self.threshold

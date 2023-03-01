class PlateauSwitchCondition:
    def __init__(self, gens=5, threshold=0.01):
        super().__init__()
        self.fitness_history = list()
        self.gens = gens
        self.threshold = threshold

    def should_approximate(self, evaluator):
        # Check if the best fitness has changed
        curr_fitness = evaluator.best_in_gen.get_pure_fitness()

        # Don't terminate during the first gens generations
        if self.gens > evaluator.gen or len(self.fitness_history) < self.gens:
            self.fitness_history.append(curr_fitness)
            return False

        del self.fitness_history[0]
        self.fitness_history.append(curr_fitness)
        return (max(self.fitness_history) - min(self.fitness_history)) <= self.threshold

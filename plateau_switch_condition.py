class PlateauSwitchCondition:
    def __init__(self, gens=5, threshold=0.01, switch_once=False):
        super().__init__()
        self.fitness_history = list()
        self.gens = gens
        self.threshold = threshold

        self.switch_once = switch_once
        if switch_once:
            self.in_plateau = False

    def should_approximate(self, evaluator):
        # If switch_once is enabled and the switch already occured, always approximate
        if self.switch_once and self.in_plateau:
            return True
        
        # Check if the best fitness has changed
        curr_fitness = evaluator.best_in_gen.get_pure_fitness()

        # Don't terminate during the first gens generations
        if self.gens > evaluator.gen or len(self.fitness_history) < self.gens:
            self.fitness_history.append(curr_fitness)
            return False

        del self.fitness_history[0]
        self.fitness_history.append(curr_fitness)
        result = (max(self.fitness_history) - min(self.fitness_history)) <= self.threshold

        if self.switch_once:
            self.in_plateau = result

        return result


from eckity.evaluators.simple_population_evaluator import SimplePopulationEvaluator


class RewardsPopulationEvaluator(SimplePopulationEvaluator):
    def _evaluate(self, population):
        res = super()._evaluate(population)
        # set rewards
        ind_eval = population.sub_populations[0].evaluator
        for individual in population.sub_populations[0].individuals:
            individual.set_rewards(ind_eval.ids2rewards[individual.id])
        return res

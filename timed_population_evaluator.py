from eckity.evaluators.simple_population_evaluator import SimplePopulationEvaluator
from time import process_time


class TimedPopulationEvaluator(SimplePopulationEvaluator):
    def __init__(self):
        super().__init__()
        self.evaluation_time = 0

    def _evaluate(self, population):
        eval_start_time = process_time()
        res = super()._evaluate(population)
        eval_end_time = process_time()
        self.evaluation_time += eval_end_time - eval_start_time
        return res
from sys import stdout
import matplotlib.pyplot as plt
import numpy as np

from eckity.statistics.statistics import Statistics
from eckity.evaluators.individual_evaluator import IndividualEvaluator


class ApproxStatistics(Statistics):
    """
    Concrete Statistics class.
    Provides statistics about the best fitness, average fitness and worst fitness of every sub-population in
    some generation.

    Parameters
    ----------
    format_string: str
        String format of the data to output.
        Value depends on the information the statistics provides.
        For more information, check out the concrete classes who extend this class.

    output_stream: Optional[SupportsWrite[str]], default=stdout
        Output file for the statistics.
        By default, the statistics will be written to stdout.
    """
    def __init__(self, ind_eval: IndividualEvaluator, format_string=None, output_stream=stdout):
        if format_string is None:
            format_string = 'best fitness {}\nworst fitness {}\naverage fitness {}\n'
        self.ind_eval = ind_eval
        self.mean_fitnesses = []
        self.median_fitnesses = []
        self.max_fitnesses = []
        self.min_fitnesses = []
        self.mean_approx_fitnesses = []
        self.median_approx_fitnesses = []
        self.max_approx_fitnesses = []
        self.min_approx_fitnesses = []
        super().__init__(format_string, output_stream)

    def write_statistics(self, sender, data_dict):
        sub_pop = data_dict['population'].sub_populations[0]
        #approx fitness
        fitnesses = np.array([ind.get_pure_fitness() for ind in sub_pop.individuals])
        self.mean_approx_fitnesses.append(np.mean(fitnesses))
        self.median_approx_fitnesses.append(np.median(fitnesses))
        self.max_approx_fitnesses.append(np.max(fitnesses))
        self.min_approx_fitnesses.append(np.min(fitnesses))
        #real fitness
        for ind in sub_pop.individuals:
            ind.set_fitness_not_evaluated()
            self.ind_eval.evaluate(ind, [])
        fitnesses = np.array([ind.get_pure_fitness() for ind in sub_pop.individuals])
        self.mean_fitnesses.append(np.mean(fitnesses))
        self.median_fitnesses.append(np.median(fitnesses))
        self.max_fitnesses.append(np.max(fitnesses))
        self.min_fitnesses.append(np.min(fitnesses))

    # TODO tostring to indiv

    def plot_statistics(self):
        assert len(self.mean_fitnesses) == len(self.median_fitnesses) == \
               len(self.max_fitnesses) == len(self.min_fitnesses) == \
               len(self.mean_approx_fitnesses) == len(self.median_approx_fitnesses) == \
               len(self.max_approx_fitnesses) == len(self.min_approx_fitnesses), \
               'Statistics lists are not the same length'

        plt.plot(self.mean_approx_fitnesses, label='approx mean')
        plt.plot(self.median_approx_fitnesses, label='approx median')
        plt.plot(self.max_approx_fitnesses, label='approx max')
        plt.plot(self.min_approx_fitnesses, label='approx min')
        plt.plot(self.mean_fitnesses, label='mean')
        plt.plot(self.median_fitnesses, label='median')
        plt.plot(self.max_fitnesses, label='max')
        plt.plot(self.min_fitnesses, label='min')
        plt.xlabel('generation')
        plt.ylabel('fitness')
        plt.xticks(range(0, len(self.mean_fitnesses) + 1, 5))
        plt.legend()
        plt.show()

    # Necessary for valid pickling, since modules cannot be pickled
    def __getstate__(self):
        state = self.__dict__.copy()
        del state['output_stream']
        return state

    # Necessary for valid unpickling, since modules cannot be pickled
    def __setstate__(self, state):
        self.__dict__.update(state)
        self.output_stream = stdout

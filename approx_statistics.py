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
        self.mean_approx_fitnesses = []
        self.median_approx_fitnesses = []
        self.max_approx_fitnesses = []
        self.mean_alternative_fitnesses = []
        self.median_alternative_fitnesses = []
        self.max_alternative_fitnesses = []
        super().__init__(format_string, output_stream)

    def write_statistics(self, sender, data_dict):
        sub_pops = data_dict['population'].sub_populations
        pop_eval = sender.population_evaluator

        #approx fitness
        approx_fitnesses = np.array([ind.get_pure_fitness() for ind in sub_pops[0].individuals])
        self.mean_approx_fitnesses.append(np.mean(approx_fitnesses))
        self.median_approx_fitnesses.append(np.median(approx_fitnesses))
        self.max_approx_fitnesses.append(np.max(approx_fitnesses))

        #real fitness
        fitnesses = np.array(pop_eval._evaluate_individuals(
            sub_pops[0].individuals,
            self.ind_eval,
            sub_population_idx=0,
            sample=True)
        )
        self.mean_fitnesses.append(np.mean(fitnesses))
        self.median_fitnesses.append(np.median(fitnesses))
        self.max_fitnesses.append(np.max(fitnesses))

        # alternative fitness
        pop_eval = sender.population_evaluator
        if pop_eval.split_experiment and pop_eval.approx_count > 0:
            fitnesses = np.array(pop_eval._evaluate_individuals(
                sub_pops[1].individuals,
                self.ind_eval,
                sub_population_idx=1,
                sample=True)
            )
            self.mean_alternative_fitnesses.append(np.mean(fitnesses))
            self.median_alternative_fitnesses.append(np.median(fitnesses))
            self.max_alternative_fitnesses.append(np.max(fitnesses))

        else:
            # no split experiment
            if pop_eval.is_approx:   
                fitnesses = np.array(pop_eval._evaluate_individuals(
                sub_pops[0].individuals,
                self.ind_eval,
                sub_population_idx=0,
                sample=True)
                )
                self.mean_fitnesses.append(np.mean(fitnesses))
                self.median_fitnesses.append(np.median(fitnesses))
                self.max_fitnesses.append(np.max(fitnesses))

            else:
                self.mean_fitnesses.append(np.mean(approx_fitnesses))
                self.median_fitnesses.append(np.median(approx_fitnesses))
                self.max_fitnesses.append(np.max(approx_fitnesses))
                
    def plot_statistics(self):
        print('mean_approx_fitnesses =', self.mean_approx_fitnesses)
        print('median_approx_fitnesses =', self.median_approx_fitnesses)
        print('max_approx_fitnesses =', self.max_approx_fitnesses)
        print('mean_fitnesses =', self.mean_fitnesses)
        print('median_fitnesses =', self.median_fitnesses)
        print('max_fitnesses =', self.max_fitnesses)
        print('mean_alternative_fitnesses =', self.mean_alternative_fitnesses)
        print('median_alternative_fitnesses =', self.median_alternative_fitnesses)
        print('max_alternative_fitnesses =', self.max_alternative_fitnesses)

        # plt.title(f'{dsname} {model_type.__name__} {model_params}')
        # plt.plot(self.mean_approx_fitnesses, label='approx mean')
        # plt.plot(self.median_approx_fitnesses, label='approx median')
        # plt.plot(self.max_approx_fitnesses, label='approx max')
        # plt.plot(self.mean_fitnesses, label='mean')
        # plt.plot(self.median_fitnesses, label='median')
        # plt.plot(self.max_fitnesses, label='max')
        # plt.xlabel('generation')
        # plt.ylabel('fitness')
        # plt.xticks(range(0, len(self.mean_fitnesses) + 1, 5))

        # Put a legend below current axis
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
                  fancybox=True, shadow=True, ncol=5)
        plt.tight_layout()
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

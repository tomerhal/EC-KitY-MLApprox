from mlxtend.evaluate import permutation_test
import pandas as pd
import numpy as np
import sys

if len(sys.argv) < 2:
    print('Usage: python3 perm_tests.py <csv_file>')
    exit(1)

file = sys.argv[1]


def to_seconds(time_str):
    hours, minutes, seconds = time_str.split(':')
    return int(hours) * 3600 + int(minutes) * 60 + int(seconds)


def get_results(df, sample_rate=1):
    fitness_col = f'Unnamed: {str(int(float(sample_rate) * 40 - 2))}'

    fitness_scores = df[fitness_col].to_numpy()[1: 21]
    fitness_scores = fitness_scores.astype(float)
    return fitness_scores


df = pd.read_csv(file)

evo_rate = 1.0
evo_fitness = get_results(df, evo_rate)
print('Evo Fitness:', evo_fitness)
scores = {'Evo': evo_fitness}

evo_mean = np.mean(evo_fitness)
evo_median = np.median(evo_fitness)
print('Evo Mean fitness:', evo_mean)
print('Evo Median fitness:', evo_median)
rates = [str(round(rate, 1)) for rate in np.arange(0.2, 1, 0.2)]

for rate in rates:
    print('rate', rate)
    evoml_fitness = get_results(df, rate)

    if len(evoml_fitness) == 0:
        continue

    scores['EvoML'] = evoml_fitness
    print('EvoML Fitness:', scores['EvoML'])

    evoml_mean = np.mean(evoml_fitness)
    print('EvoML Mean fitness:', evoml_mean)
    means = sorted([('Evo', evo_mean), ('EvoML', evoml_mean)], key=lambda x: x[1], reverse=True)
    print('sorted means:', means)
    p_value = permutation_test(scores[means[0][0]], scores[means[1][0]], method='approximate', num_rounds=10_000,
                                    func=lambda x, y: np.abs(np.mean(x) - np.mean(y)))
    print('p-value (mean fitness):', p_value)
    if means[0][0] == 'EvoML' and p_value < 0.05:
        print('EvoML ranked first, statistically significant')
    elif means[1][0] == 'EvoML' and p_value >= 0.05:
        print('EvoML ranked second, statistically insignificant')

    evoml_median = np.median(evoml_fitness)
    print('EvoML Median fitness:', evoml_median)
    medians = sorted([('Evo', evo_median), ('EvoML', evoml_median)], key=lambda x: x[1], reverse=True)
    print('sorted medians:', medians)
    p_value = permutation_test(scores[medians[0][0]], scores[medians[1][0]], method='approximate', num_rounds=10_000,
                                    func=lambda x, y: np.abs(np.median(x) - np.median(y)))
    print('p-value (median fitness):', p_value)
    if medians[0][0] == 'EvoML' and p_value < 0.05:
        print('EvoML ranked first, statistically significant')
    elif medians[1][0] == 'EvoML' and p_value >= 0.05:
        print('EvoML ranked second, statistically insignificant')

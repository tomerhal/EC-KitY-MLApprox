import numpy as np

DATASETS = ['adult', 'magic', 'coil2000', 'agaricus_lepiota', 'mushroom', 'ring']
thresholds = {
    'adult': 0.1,
    'magic': 0.07,
    'coil2000': 0.03,
    'agaricus_lepiota': 0.04,
    'mushroom': 0.05,
    'ring': 0.02
}

linear_gen_weight = lambda gen: gen + 1
square_gen_weight = lambda gen: (gen + 1) ** 2
exp_gen_weight = lambda gen: np.e ** (gen + 1)
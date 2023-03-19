import numpy as np

BIG_DATASETS = ['adult', 'magic', 'coil2000', 'agaricus_lepiota', 'mushroom', 'ring']

# 10 datasets of 200-350 samples each
SMALL_DATASETS = ['prnn_crabs', 'sonar', 'biomed', 'heart_statlog', 'spect', 'breast_cancer', 'heart_h', 'cleve', 'bupa']

thresholds = {
    'adult': 0.1,
    'magic': 0.07,
    'coil2000': 0.03,
    'agaricus_lepiota': 0.04,
    'mushroom': 0.06,
    'ring': 0.02
}

linear_gen_weight = lambda gen: gen + 1
square_gen_weight = lambda gen: (gen + 1) ** 2
exp_gen_weight = lambda gen: np.e ** (gen + 1)
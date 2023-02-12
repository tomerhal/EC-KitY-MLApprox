"""
Test dynamic model creation and usage.
"""
import pytest

from sklearn.linear_model import SGDRegressor
from approx_ml_pop_eval import ApproxMLPopulationEvaluator


class TestDynamicModel:
	def test_sgd(self):
		X, y = [[1, 2]], [3]
		pop_eval = ApproxMLPopulationEvaluator(model_type=SGDRegressor, model_params={'max_iter': 1000})
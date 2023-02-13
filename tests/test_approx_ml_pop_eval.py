"""
Test dynamic model creation and usage.
"""
import pytest
import random
import numpy as np

from sklearn.linear_model import SGDRegressor, LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from eckity.genetic_encodings.ga.vector_individual import Vector
from eckity.fitness.simple_fitness import SimpleFitness

import sys
sys.path.append('..')
from approx_ml_pop_eval import ApproxMLPopulationEvaluator

models_params = {
	SGDRegressor: {'max_iter': 1000, 'tol': 1e-3, 'random_state': 0},
	LinearRegression: {},
	KNeighborsRegressor: {'n_neighbors': 3}
}

class TestApproxMLPopulationEvaluator:
	def _generate_vectors(self, n, bounds, length):
		vecs = [Vector(SimpleFitness(random.random()), bounds) for i in range(n)]
		a, b = bounds
		for ind in vecs:
			ind.set_vector([random.randint(a, b) for _ in range(length)])
		return vecs
	
	def test_model(self):
		X_train = self._generate_vectors(10, (0, 9), 2)
		y_train = [ind.get_pure_fitness() for ind in X_train]
		
		X_test = self._generate_vectors(2, (10, 19), 2)
		y_test = [ind.fitness.fitness for ind in X_test]

		for model_type, params in models_params.items():
			pop_eval = ApproxMLPopulationEvaluator(model_type=model_type, model_params=params)
			pop_eval.fit(X_train, y_train)

			model = model_type(**params)
			V_train = np.array([ind.vector for ind in X_train])
			model.fit(V_train, y_train)

			V_test = np.array([ind.vector for ind in X_test])
			assert model.score(V_test, y_test) == pop_eval.model.score(V_test, y_test)

	def test_accumulate_population_fitness(self):
		for model_type, params in models_params.items():
			pop_eval = ApproxMLPopulationEvaluator(
				model_type=model_type,
				model_params=params,
				accumulate_population_data=True,
				cache_fitness=True
			)
			individuals = self._generate_vectors(10, (0, 9), 2)
			fitnesses = [ind.get_pure_fitness() for ind in individuals]
			pop_eval.fit(individuals, fitnesses)
			for ind, fitness in zip(individuals, fitnesses):
				assert ind.vector + [fitness] in pop_eval.df.values
	
	def test_predict(self):
		for metric in [mean_absolute_error, mean_squared_error, r2_score]:
			for model_type, params in models_params.items():
				pop_eval = ApproxMLPopulationEvaluator(
					model_type=model_type,
					model_params=params,
					scoring=metric
				)
				X_train = self._generate_vectors(10, (0, 9), 2)
				y_train = [ind.get_pure_fitness() for ind in X_train]				
				X_test = self._generate_vectors(2, (10, 19), 2)
				y_test = [ind.fitness.fitness for ind in X_test]

				pop_eval.fit(X_train, y_train)

				model = model_type(**params)
				V_train = np.array([ind.vector for ind in X_train])
				V_test = np.array([ind.vector for ind in X_test])
				model.fit(V_train, y_train)
				
				assert metric(y_test, model.predict(V_test)) == metric(y_test, pop_eval.predict(X_test))

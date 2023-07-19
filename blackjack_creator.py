from random import random

from eckity.creators.ga_creators.simple_vector_creator import GAVectorCreator
from blackjack_individual import BlackjackIndividual

class BlackjackVectorCreator(GAVectorCreator):
	def __init__(self,
				length=1,
				gene_creator=None,
				bounds=(0.0, 1.0),
				events=None,
				original_vector=None,
				noisy=False,
				noise=0.1):

		if noisy:
			self.original_vector = original_vector
			self.noise = noise
			gene_creator = lambda _, index: 1 - original_vector[index] if random() < noise else original_vector[index]

		super().__init__(length=length,
						 gene_creator=gene_creator,
						 bounds=bounds,
						 vector_type=BlackjackIndividual,
						 events=events)

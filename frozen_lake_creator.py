from random import random, choice

from eckity.creators.ga_creators.simple_vector_creator import GAVectorCreator
from frozen_lake_individual import FrozenLakeIndividual

class FrozenLakeVectorCreator(GAVectorCreator):
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
			# with probability noise, replace the gene with a random other gene
            # with probability 1 - noise, keep the original gene
			gene_creator = lambda _, index: choice(list(set(range(bounds[0], bounds[1] + 1)) - {original_vector[index]})) \
		                                    if random() < noise \
						                    else original_vector[index]

		super().__init__(length=length,
						 gene_creator=gene_creator,
						 bounds=bounds,
						 vector_type=FrozenLakeIndividual,
						 events=events)

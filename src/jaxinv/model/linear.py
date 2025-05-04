"""linear models"""

import jax.numpy as jnp

def type1(geometric_weight, map_vector):
    return jnp.dot(geometric_weight, map_vector)

def type2(geometric_weight, spectral_matrix, map_matrix):
    return geometric_weight @ map_matrix @ spectral_matrix

def type3(geometric_weight, map_vector):
    return jnp.sum(geometric_weight*map_vector, axis=1)
from jaxinv.bayes.weighted_kernel import generalized_weighted_kernel_type1
from jaxinv.bayes.weighted_kernel import generalized_weighted_kernel_type2
from jaxinv.bayes.weighted_kernel import generalized_weighted_kernel_type3
import jax.numpy as jnp
from jax import random

def test_generalized_weighted_kernel_type1():
    """Test the generalized weighted kernel of type 1"""
    Ni = 2
    Nj = 3
    key = random.PRNGKey(0)
    geometric_weight = random.normal(key, shape=(Ni, Nj))
    kernel_model = jnp.diag(random.uniform(key, shape=(Nj,)))

    result = generalized_weighted_kernel_type1(geometric_weight, kernel_model)
    expected_result = jnp.array([[6.571487,   0.36815816], [0.36815816, 0.35020235]])

    assert jnp.allclose(result, expected_result), f"Expected {expected_result}, got {result}"


if __name__ == "__main__":
    test_generalized_weighted_kernel_type1()
    
from jaxinv.bayes.weighted_kernel import generalized_weighted_kernel_type1
from jaxinv.bayes.weighted_kernel import generalized_weighted_kernel_type2
from jaxinv.bayes.weighted_kernel import generalized_weighted_kernel_type3
import pytest
import jax.numpy as jnp
from jax import random


def test_generalized_weighted_kernel_type1():
    """Test the generalized weighted kernel of type 1"""
    geometric_weight, _, kernel_model, _, _ = _test_kernel_model(2, 3, seed=0)

    result = generalized_weighted_kernel_type1(geometric_weight, kernel_model)
    expected_result = jnp.array([[6.571487, 0.36815816], [0.36815816, 0.35020235]])

    assert jnp.allclose(
        result, expected_result
    ), f"Expected {expected_result}, got {result}"


def test_generalized_weighted_kernel_type2():
    """Test the generalized weighted kernel of type 2"""
    geometric_weight, spectral_matrix, kernel_model_s, _, kernel_model_x = (
        _test_kernel_model(seed=0)
    )
    result = generalized_weighted_kernel_type2(
        geometric_weight, spectral_matrix, kernel_model_s, kernel_model_x
    )

    assert result.shape == (10, 10), f"Expected shape (10, 10), got {result.shape}"
    assert (
        jnp.sum(result) == pytest.approx(106.90668)
    ), f"Expected sum 106.90668, got {jnp.sum(result)}"


def test_generalized_weighted_kernel_type3():
    """Test the generalized weighted kernel of type 3"""
    geometric_weight, _, kernel_model_s, kernel_model_t, _ = _test_kernel_model(seed=0)
    result = generalized_weighted_kernel_type3(
        geometric_weight, kernel_model_s, kernel_model_t
    )
    
    
    assert result.shape == (2, 2), f"Expected shape (2, 2), got {result.shape}"
    assert jnp.sum(result) == pytest.approx(6.5702825), f"Expected sum 6.5702825, got {jnp.sum(result)}"


def _test_kernel_model(Ni=2, Nj=3, Nk=4, Nl=5, seed=0):
    key = random.PRNGKey(seed)
    geometric_weight = random.normal(key, shape=(Ni, Nj))
    spectral_matrix = random.normal(key, shape=(Nk, Nl))
    kernel_model_t = jnp.diag(random.uniform(key, shape=(Ni,)))
    kernel_model_s = jnp.diag(random.uniform(key, shape=(Nj,)))
    kernel_model_x = jnp.diag(random.uniform(key, shape=(Nk,)))

    return (
        geometric_weight,
        spectral_matrix,
        kernel_model_s,
        kernel_model_t,
        kernel_model_x,
    )


if __name__ == "__main__":
    test_generalized_weighted_kernel_type1()
    test_generalized_weighted_kernel_type2()
    test_generalized_weighted_kernel_type3()
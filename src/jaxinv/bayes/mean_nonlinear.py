import jax.numpy as jnp
from jax.scipy.linalg import solve
from jaxinv.bayes.weighted_kernel import generalized_weighted_kernel_type1


def mean_nonlinear_type1(geometric_weight, kernel_model, precision_matrix_data, data):

    Kw = generalized_weighted_kernel_type1(geometric_weight, kernel_model)
    Nn = jnp.shape(Kw)[0]
    IKw = jnp.eye(Nn) + precision_matrix_data @ Kw
    Xlc = solve(IKw, precision_matrix_data @ data, assume_a="pos")

    return kernel_model @ geometric_weight.T @ Xlc

import jax.numpy as jnp
from jax.scipy.linalg import solve
from jaxinv.bayes.weighted_kernel import generalized_weighted_kernel_type1
from jaxinv.bayes.weighted_kernel import generalized_weighted_kernel_type3


def meanmap_type1(geometric_weight, kernel_model, precision_matrix_data, data):

    Kw = generalized_weighted_kernel_type1(geometric_weight, kernel_model)
    Nn = jnp.shape(Kw)[0]
    IKw = jnp.eye(Nn) + precision_matrix_data @ Kw
    Xlc = solve(IKw, precision_matrix_data @ data, assume_a="pos")

    return kernel_model @ geometric_weight.T @ Xlc


def meanmap_type3(
    geometric_weight, kernel_model_s, kernel_model_t, alpha, precision_matrix_data, data
):
    Ni, _ = jnp.shape(geometric_weight)
    Kw = generalized_weighted_kernel_type3(
        geometric_weight, kernel_model_s, kernel_model_t, alpha
    )
    IKw = jnp.eye(Ni) + precision_matrix_data @ Kw
    Xlc = solve(IKw, precision_matrix_data @ data)
    return alpha * kernel_model_t @ (geometric_weight.T * Xlc).T @ kernel_model_s

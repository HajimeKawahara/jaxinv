import jax.numpy as jnp
from jax.scipy.linalg import solve
from jaxinv.bayes.weighted_kernel import generalized_weighted_kernel_type1
from jaxinv.bayes.weighted_kernel import generalized_weighted_kernel_type2
from jaxinv.bayes.weighted_kernel import generalized_weighted_kernel_type3


def meanmap_type1(geometric_weight, kernel_model, precision_matrix_data, data):

    Kw = generalized_weighted_kernel_type1(geometric_weight, kernel_model)
    Nn = jnp.shape(Kw)[0]
    IKw = jnp.eye(Nn) + precision_matrix_data @ Kw
    nvector = solve(IKw, precision_matrix_data @ data, assume_a="pos")

    return kernel_model @ geometric_weight.T @ nvector


def meanmap_type2(
    geometric_weight,
    spectral_matrix,
    kernel_model_s,
    kernel_model_x,
    alpha,
    precision_matrix_data,
    data,
):

    Ni, _ = jnp.shape(geometric_weight)
    _, Nl = jnp.shape(spectral_matrix)
    Ky = generalized_weighted_kernel_type2(
        geometric_weight, spectral_matrix, kernel_model_s, kernel_model_x
    )
    IKw = jnp.eye(Ni) + alpha * precision_matrix_data @ Ky
    nvector = solve(IKw, precision_matrix_data @ data)
    nmatrix = nvector.reshape((Ni, Nl))

    return (
        kernel_model_s
        @ geometric_weight.T
        @ nmatrix
        @ spectral_matrix.T
        @ kernel_model_x
    )


def meanmap_type3(
    geometric_weight, kernel_model_s, kernel_model_t, alpha, precision_matrix_data, data
):
    Ni, _ = jnp.shape(geometric_weight)
    Kw = generalized_weighted_kernel_type3(
        geometric_weight, kernel_model_s, kernel_model_t
    )
    IKw = jnp.eye(Ni) + alpha * precision_matrix_data @ Kw
    nvector = solve(IKw, precision_matrix_data @ data)
    
    #original algorithm from sot rundynamic_cpu
    return alpha * kernel_model_t @ (geometric_weight.T * nvector).T @ kernel_model_s 
    #return alpha * (kernel_model_t @ geometric_weight @ kernel_model_s ).T @ nvector

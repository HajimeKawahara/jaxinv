import jax.numpy as jnp
from jax.scipy.linalg import solve
from jaxinv.bayes.weighted_kernel import generalized_weighted_kernel_type1
from jaxinv.bayes.weighted_kernel import generalized_weighted_kernel_type2
from jaxinv.bayes.weighted_kernel import generalized_weighted_kernel_type3
from jaxinv.bayes.weighted_kernel import generalized_weighted_kernel_type4


def meanmap_inverse_type1(geometric_weight, kernel_model, precision_matrix_data, data):
    """computes mean map using the inverse-mode computation for the type 1 kernel
    
    Args:
        geometric_weight (2D array): Geometric weight matrix (Ni, Nj).
        kernel_model (2D array): spatial Kernel (Nj, Nj).
        precision_matrix_data (2D array): Precision matrix of the data (Ni, Ni).
        data (1D array): Data matrix (Ni)

    Returns:
       1D array: Mean map (Nj) 

    """

    Kw = generalized_weighted_kernel_type1(geometric_weight, kernel_model)
    Nn = jnp.shape(Kw)[0]
    IKw = jnp.eye(Nn) + precision_matrix_data @ Kw
    nvector = solve(IKw, precision_matrix_data @ data, assume_a="pos")

    return kernel_model @ geometric_weight.T @ nvector


def meanmap_inverse_type2(
    geometric_weight,
    spectral_matrix,
    kernel_model_s,
    kernel_model_x,
    alpha,
    precision_matrix_data,
    data,
):
    """computes mean map using the inverse-mode computation for the type 2 kernel

    Notes:
        See tests/unittests/math/kronflatten_test.py for the Kron/flatten/reshape operations

    Args:
        geometric_weight (2D array): Geometric weight matrix (Ni, Nj).
        spectral_matrix (2D array): Spectral matrix (Nk, Nl).
        kernel_model_s (2D array): spatial kernel (Nj, Nj)
        kernel_model_x (2D array): spectral kernel (Nk, Nk)
        alpha (float): GP normalization factor.
        precision_matrix_data (2D array): Precision matrix of the data (Ni*Nl, Ni*Nl).
        data (2D array): Data matrix (Ni, Nl)
    
    Returns:
        2D array: Mean map (Nj, Nk).

    """
    Ni, _ = jnp.shape(geometric_weight)
    _, Nl = jnp.shape(spectral_matrix)
    Ky = generalized_weighted_kernel_type2(
        geometric_weight, spectral_matrix, kernel_model_s, kernel_model_x
    )
    I_plus_KG = jnp.eye(Ni * Nl) + alpha * precision_matrix_data @ Ky
    nvector = solve(I_plus_KG, precision_matrix_data @ data.flatten())
    nmatrix = nvector.reshape((Ni, Nl))

    return (
        alpha
        * kernel_model_s
        @ geometric_weight.T
        @ nmatrix
        @ spectral_matrix.T
        @ kernel_model_x
    )


def meanmap_inverse_type3(
    geometric_weight, kernel_model_s, kernel_model_t, alpha, precision_matrix_data, data
):
    Ni, _ = jnp.shape(geometric_weight)
    Kw = generalized_weighted_kernel_type3(
        geometric_weight, kernel_model_s, kernel_model_t
    )
    IKw = jnp.eye(Ni) + alpha * precision_matrix_data @ Kw
    nvector = solve(IKw, precision_matrix_data @ data)

    # original algorithm from sot rundynamic_cpu
    return alpha * kernel_model_t @ (geometric_weight.T * nvector).T @ kernel_model_s
    # return alpha * (kernel_model_t @ geometric_weight @ kernel_model_s ).T @ nvector


def meanmap_inverse_type4(
    geometric_weight,
    spectral_matrix,
    kernel_model_s,
    kernel_model_t,
    kernel_model_x,
    alpha,
    precision_matrix_data,
    data,
):
    Ni, _ = jnp.shape(geometric_weight)
    _, Nl = jnp.shape(spectral_matrix)
    Ky = generalized_weighted_kernel_type4(
        geometric_weight,
        spectral_matrix,
        kernel_model_s,
        kernel_model_t,
        kernel_model_x,
    )
    IKw = jnp.eye(Ni) + alpha * precision_matrix_data @ Ky
    nvector = solve(IKw, precision_matrix_data @ data)
    nmatrix = nvector.reshape((Ni, Nl))

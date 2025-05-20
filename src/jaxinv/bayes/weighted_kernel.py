import jax.numpy as jnp


def generalized_weighted_kernel_type1(geometric_weight, kernel_model):
    """computes the generalized weighted kernel of type 1

    Args:
        geometic_weight (2D array): geometric weight (Ni, Nj)
        kernel_model (2D array): kernel for the model covariances (Nj, Nj)

    Returns:
        2D array: generalized weighted kernel of type 1 (Ni, Ni)
    """
    return geometric_weight @ kernel_model @ geometric_weight.T


def weighted_spectral_kernel(spectral_matrix, kernel_model_x):
    """computes the weighted spectral kernel
    Args:
        spectral_matrix (2D array): component -- spectral matrix, X (Nk, Nl)
        kernel_model_x (2D array): kernel for the model covariances for K( components) (Nk, Nk)
    Returns:
        2D array: weighted spectral kernel (Nl, Nl)
    """

    return spectral_matrix.T @ kernel_model_x @ spectral_matrix


def generalized_weighted_kernel_type2(
    geometric_weight, spectral_matrix, kernel_model_s, kernel_model_x
):
    """computes the generalized weighted kernel of type 2

    Args:
        geometic_weight (2D array): geometric weight, W (Ni, Nj)
        spectral_matrix (2D array): component -- spectral matrix, X (Nk, Nl)
        kernel_model_s (2D array): kernel for the model covariances for S(patial) (Nj, Nj)
        kernel_model_x (2D array): kernel for the model covariances for K( components) (Nk, Nk)

    Returns:
        2D array: generalized weighted kernel of type 2 (Ni*Nl, Ni*Nl)
    """
    wkernel_type1 = generalized_weighted_kernel_type1(geometric_weight, kernel_model_s)
    wkernel_spectral = weighted_spectral_kernel(spectral_matrix, kernel_model_x)
    return jnp.kron(wkernel_type1, wkernel_spectral)


def generalized_weighted_kernel_type3(geometric_weight, kernel_model_s, kernel_model_t):
    """computes the generalized weighted kernel of type 3

    Args:
        geometic_weight (2D array): geometric weight (Ni, Nj)
        kernel_model_s (2D array): kernel for the model covariances for S(patial) (Nj, Nj)
        kernel_model_t (2D array): kernel for the model covariances for T(emporal) (Ni, Ni)

    Returns:
        2D array: generalized weighted kernel of type 2 (Ni, Ni)
    """
    wkernel_type1 = generalized_weighted_kernel_type1(geometric_weight, kernel_model_s)
    return wkernel_type1 * kernel_model_t


def generalized_weighted_kernel_type4(
    geometric_weight, spectral_matrix, kernel_model_s, kernel_model_t, kernel_model_x
):
    """computes the generalized weighted kernel of type 4

    Args:
        geometric_weight (2D array): geometric weight, W (Ni, Nj)
        spectral_matrix (2D array): component -- spectral matrix, X (Nk, Nl)
        kernel_model_s (2D array): kernel for the model covariances for S(patial) (Nj, Nj)
        kernel_model_t (2D array): kernel for the model covariances for T(emporal) (Ni, Ni)
        kernel_model_x (2D array): kernel for the model covariances for K( components) (Nk, Nk)

    Returns:
        2D array: generalized weighted kernel of type 4 (Ni*Nl, Ni*Nl)
    """
    wkernel_spectral = weighted_spectral_kernel(spectral_matrix, kernel_model_x)
    kernel_type3 = generalized_weighted_kernel_type3(
        geometric_weight, kernel_model_s, kernel_model_t
    )
    return jnp.kron(wkernel_spectral, kernel_type3)

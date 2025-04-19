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


def generalized_weighted_kernel_type2(
    geometric_weight, spectral_matrix, kernel_model_s, kernel_model_x, alpha
):
    """computes the generalized weighted kernel of type 2

    Args:
        geometic_weight (2D array): geometric weight, W (Ni, Nj)
        spectral_matrix (2D array): component -- spectrum matrix, X (Nk, Nl)
        kernel_model_s (2D array): kernel for the model covariances for S(patial) (Nj, Nj)
        kernel_model_x (2D array): kernel for the model covariances for K( components) (Nk, Nk)
        alpha (float): intensity of the kernel

    Returns:
        2D array: generalized weighted kernel of type 2 (Ni*Nl, Ni*Nl)
    """
    WKW = geometric_weight @ kernel_model_s @ geometric_weight.T
    XKX = spectral_matrix.T @ kernel_model_x @ spectral_matrix
    return alpha * jnp.kron(WKW, XKX)


def generalized_weighted_kernel_type3(
    geometric_weight, kernel_model_s, kernel_model_t, alpha
):
    """computes the generalized weighted kernel of type 3

    Args:
        geometic_weight (2D array): geometric weight (Ni, Nj)
        kernel_model_s (2D array): kernel for the model covariances for S(patial) (Nj, Nj)
        kernel_model_t (2D array): kernel for the model covariances for T(emporal) (Ni, Ni)
        alpha (float): intensity of the kernel

    Returns:
        2D array: generalized weighted kernel of type 2 (Ni, Ni)
    """
    return (
        alpha
        * kernel_model_t
        * (geometric_weight @ kernel_model_s @ geometric_weight.T)
    )


def generalized_weighted_kernel_type4(
    geometric_weight,
    spectral_matrix,
    kernel_model_s,
    kernel_model_t,
    kernel_model_x,
    alpha,
):
    """computes the generalized weighted kernel of type 4

    Args:
        geometic_weight (2D array): geometric weight, W (Ni, Nj)
        spectral_matrix (2D array): component -- spectrum matrix, X (Nk, Nl)
        kernel_model_s (2D array): kernel for the model covariances for S(patial) (Nj, Nj)
        kernel_model_t (2D array): kernel for the model covariances for T(emporal) (Ni, Ni)
        kernel_model_x (2D array): kernel for the model covariances for K( components) (Nk, Nk)
        alpha (float): intensity of the kernel

    Returns:
        2D array: generalized weighted kernel of type 4 (Ni, Ni)
    """
    XKX = spectral_matrix.T @ kernel_model_x @ spectral_matrix
    WKW = geometric_weight @ kernel_model_s @ geometric_weight.T
    return alpha * jnp.kron(WKW * kernel_model_t, XKX)

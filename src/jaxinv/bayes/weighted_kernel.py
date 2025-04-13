def generalized_weighted_kernel_type1(geometric_weight, kernel_model):
    """computes the generalized weighted kernel of type 1

    Args:
        geometic_weight (2D array): geometric weight (Ni, Nj)
        kernel_model (2D array): kernel for the model covariances (Nj, Nj)

    Returns:
        2D array: generalized weighted kernel of type 1 (Ni, Ni)
    """
    return geometric_weight @ kernel_model @ geometric_weight.T


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

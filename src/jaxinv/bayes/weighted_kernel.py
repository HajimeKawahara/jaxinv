
def generalized_weighted_kernel_type1(geometric_weight, kernel_model):
    """computes the generalized weighted kernel of type 1

    Args:
        geometic_weight (2D array): geometric weight (Ni, Nj)
        kernel_model (2D array): kernel for the model covariances (Nj, Nj)

    Returns:
        2D array: generalized weighted kernel of type 1 (Ni, Ni)
    """
    return geometric_weight@kernel_model@geometric_weight.T


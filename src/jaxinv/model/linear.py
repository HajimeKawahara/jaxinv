"""linear (in terms of map tensor) models"""

import jax.numpy as jnp


def type1(geometric_weight, map_vector):
    """type 1 linear model

    Args:
        geometric_weight (2D array): geometric weight (Ni, Nj)
        map_vector (vector): map vector (Nj,)

    Returns:
        vector: data from the linear model (Ni,)
    """
    return jnp.dot(geometric_weight, map_vector)


def type2(geometric_weight, spectral_matrix, map_matrix):
    """type 2 linear model

    Args:
        geometric_weight (2D array): geometric weight (Ni, Nj)
        spectral_matrix (2D array): spectral matrix (Nk, Nl)
        map_matrix (2D array): map matrix (Nj, Nk)

    Returns:
        2D array: data from the linear model (Ni,Nl)
    """
    return geometric_weight @ map_matrix @ spectral_matrix


def type3(geometric_weight, map_matrix):
    """type 3 linear model

    Args:
        geometric_weight (2D array): geometric weight (Ni, Nj)
        map_matrix (2D array): map matrix (Ni, Nj)

    Returns:
        vector: data from the linear model (Ni,)
    """
    return jnp.sum(geometric_weight * map_matrix, axis=1)


def type4(geometric_weight, spectral_matrix, map_tensor):
    """type 4 linear model

    Args:
        geometric_weight (2D array): geometric weight (Ni, Nj)
        spectral_matrix (2D array): spectral matrix (Nk, Nl)
        map_tensor (3D array): map tensor (Ni,Nj,Nk)

    Returns:
        2D array: data from the linear model (Ni,Nl)
    """
    return jnp.einsum("ij,ijk,kl->il", geometric_weight, map_tensor, spectral_matrix)


def type5(multi_geometric_weight, spectral_matrix, map_matrix):
    """type 5 linear model

    Args:
        multi_geometric_weight (3D array): geometric weight (Ni, Nj, Nk)
        spectral_matrix (2D array): spectral matrix (Nk, Nl)
        map_matrix (2D array): map vector (Nj,Nk)

    Returns:
        2D array: data from the linear model (Ni,Nl)
    """
    return jnp.einsum(
        "ijk,ijk,kl->il", multi_geometric_weight, map_matrix, spectral_matrix
    )


def type6(multi_geometric_weight, spectral_matrix, map_tensor):
    """type 6 linear model

    Args:
        multi_geometric_weight (3D array): geometric weight (Ni, Nj, Nk)
        spectral_matrix (2D array): spectral matrix (Nk, Nl)
        map_tensor (3D array): map tensor (Ni,Nj,Nk)

    Returns:
        2D array: data from the linear model (Ni,Nl)
    """
    return jnp.einsum(
        "ijk,ijk,kl->il", multi_geometric_weight, map_tensor, spectral_matrix
    )


def type7(spectral_geometric_weight, spectral_vector, map_vector):
    """type 7 linear model

    Args:
        spectral_geometric_weight (3D array): geometric weight (Ni, Nj, Nl)
        spectral_vector (vector): spectral vector (Nl,)
        map_vector (vector): map vector of shape (Nj,)

    Returns:
        2D array: data from the linear model (Ni,Nl)
    """
    return jnp.einsum(
        "ijl,j,l->il", spectral_geometric_weight, map_vector, spectral_vector
    )


if __name__ == "__main__":
    pass
    # type4(geometric_weight, spectral_matrix, map_tensor)

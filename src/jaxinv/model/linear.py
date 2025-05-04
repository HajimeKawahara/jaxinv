"""linear models"""

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
        map_matrix (2D array): map vector (Nj,Nk)

    Returns:
        2D array: data from the linear model (Ni,Nl) 
    """
    return geometric_weight @ map_matrix @ spectral_matrix

def type3(geometric_weight, map_matrix):
    """type 3 linear model
    
    Args:
        geometric_weight (2D array): geometric weight (Ni, Nj)
        map_matrix (vector): map matrix (Ni, Nj)

    Returns:
        vector: data from the linear model (Ni,) 
    """
    return jnp.sum(geometric_weight*map_matrix, axis=1)

def type4(geometric_weight, spectral_matrix, map_tensor):
    """type 4 linear model
    
    Args:
        geometric_weight (2D array): geometric weight (Ni, Nj)
        spectral_matrix (2D array): spectral matrix (Nk, Nl)
        map_tensor (3D array): map tensor (Ni,Nj,Nk)

    Returns:
        2D array: data from the linear model (Ni,Nl) 
    """
    #return jnp.sum(geometric_weight*map_tensor, axis=1)@spectral_matrix
    return jnp.einsum("ij,ijk,kl->il", geometric_weight, map_tensor, spectral_matrix)

if __name__ == "__main__":
    pass
    #type4(geometric_weight, spectral_matrix, map_tensor)
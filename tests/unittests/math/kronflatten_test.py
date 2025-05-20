"""Unit tests for the Kronecker product and flattening relation.

Notes:
    How to implement the relation WAX = mat( (X^T \otimes W) @ A)
    n,m = A.shape 
    1. vec(A) = A.flatten(), X^T \otimes W = kron(W, X^T), mat(a) = a.reshape(n,m) 
    2. vec(A) = A.T.flatten(), X^T \otimes W = kron(X^T, W), mat(a) = a.reshape(m,n).T
    3. vec(A) = A.flatten(order="F"), X^T \otimes W = kron(X^T, W), mat(a) = a.reshape(m,n).T
    The default strategy is to use the first one

"""


import jax.numpy as jnp


def test_kron_flatten_relation():
    W = jnp.array([[1, 4], [2, 9]])  # (Ni=2, Nj=2)
    A = jnp.array([[7, 5], [1, 8]])  # (Nj=2, Nk=2)
    X = jnp.array([[1, 2, 3], [4, 5, 6]])  # (Nk=2, Nl=3)
    WAX = W @ A @ X
    vec_WAX = jnp.kron(W, X.T) @ A.flatten()
    vec_WAX_trans = jnp.kron(X.T, W) @ A.T.flatten()
    vec_WAX_F = jnp.kron(X.T, W) @ A.flatten(order="F")
    n, m = WAX.shape
    
    assert jnp.all(WAX == vec_WAX.reshape(n, m))
    assert jnp.all(WAX == vec_WAX_trans.reshape(m, n).T)
    assert jnp.all(WAX == vec_WAX_F.reshape(m, n).T)
    

if __name__ == "__main__":
    test_kron_flatten_relation()
    print("test")

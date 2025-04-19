import jax.numpy as jnp


def rbf(dx, tau):
    return jnp.exp(-((dx) ** 2) / 2 / (tau**2))


def matern32(dx, tau):
    fac = jnp.sqrt(3.0) * jnp.abs(dx) / tau
    return (1.0 + fac) * jnp.exp(-fac)

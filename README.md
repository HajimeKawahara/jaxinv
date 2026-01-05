# jaxinv

Differentiable Bayesian Framework for Inverse Problem (using JAX).
Non-SLIP (Nonlinear-parameters Sampling in a Linear Inverse Problem) method should be applicable to all of the types in JAXINV.

JAXINV will be used in https://github.com/HajimeKawahara/sot

Set the 64-bit mode:

```python
from jax import config
config.update("jax_enable_x64", True)
```

## Types

- type 1: standard linear inverse problem (d = W a)
- type 2: unmixing (D = W A X)
- type 3: dynamic d = (W o I) a, o=face-splitting product
- type 4: dynamic unmixing
- type 5: multi-weight unmixing
- type 6: dynamic multi-weight unmixing
- type 1a: static Doppler Imaging
- type 1b: multi-spectral Doppler Imaging
- type 3a: dynamic Doppler Imaging
- type 3b: dynamic multi-spectral Doppler Imaging
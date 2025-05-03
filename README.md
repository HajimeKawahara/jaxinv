# jaxinv
Differentiable Bayesian Framework for Inverse Problem (using JAX)

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
- type 7: doppler-imaging like
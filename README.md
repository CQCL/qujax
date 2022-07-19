# quax

Represent a (parameterised) quantum circuit as a pure [JAX](https://github.com/google/jax) function that
takes as input any parameters of the circuit and outputs a _statetensor_. The statetensor encodes all $2^N$ amplitudes of the quantum state and can then be used
downstream for exact expectations, gradients or sampling.

A JAX implementation of a quantum circuit is useful for runtime speedups, automatic differentiation and support for GPUs/TPUs.

## Install
```
pip install quax
```

## Run
```python
from jax import numpy as jnp
import quax

circuit_gates = ['H', 'Rz', 'CX']
circuit_qubit_inds = [[0], [0], [0, 1]]
circuit_params_inds = [[], [0], []]

params_to_st = quax.get_params_to_statetensor_func(circuit_gates,
                                                   circuit_qubit_inds,
                                                   circuit_params_inds)
```

We now have a pure JAX function that generates the statetensor for given parameters
```python
params_to_st(jnp.array([1.3]))
# DeviceArray([[-0.32101968-0.6300368j,  0.        +0.j       ],
#              [ 0.        +0.j       , -0.32101968+0.6300368j]],            dtype=complex64)
```

The statevector can be obtained from the statetensor via ```.flatten()```.
```python
params_to_st(jnp.array([1.3])).flatten()
# DeviceArray([-0.32101976-0.63003676j,   0.+0.j,  0.+0.j,    -0.32101976+0.63003676j],            dtype=complex64)
```

## Notes
+ We can then apply a cost function (that maps the statetensor to a scalar variable) and use ```jax.grad``` to obtain exact gradients.
+ We use the convention where parameters are given in units of π (i.e. in [0,2] rather than [0, 2π]).


## quax.tket
You can also generate the parameter to statetensor function from a [pytket](https://cqcl.github.io/tket/pytket/api/) circuit
```python
import pytket
import quax

circuit = pytket.Circuit(2)
circuit.H(0)
circuit.Rz(1.3, 0)
circuit.CX(0, 1)

params_to_st = quax.tket.tk_to_quax(circuit)
```


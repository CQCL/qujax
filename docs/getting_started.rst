Getting started
#################

**Important note**: qujax circuit parameters are expressed in units of :math:`\pi` (e.g. in the range :math:`[0,2]` as opposed to :math:`[0, 2\pi]`).

*********************
Pure state simulation
*********************

We start by defining the quantum gates making up the circuit, along with the qubits that they act on and the indices of the parameters for each gate. 

A list of all gates can be found in :doc:`gates` (custom operations can be included by passing an array or function instead of a string, as documented in :doc:`statetensor/get_params_to_statetensor_func`).

.. code-block:: python

   from jax import numpy as jnp
   import qujax
   
   # List of quantum gates
   circuit_gates = ['H', 'Ry', 'CZ']
   # Indices of qubits the gates will be applied to
   circuit_qubit_inds = [[0], [0], [0, 1]]
   # Indices of parameters each parameterised gate will use
   circuit_params_inds = [[], [0], []]
   
   qujax.print_circuit(circuit_gates, circuit_qubit_inds, circuit_params_inds);
   # q0: -----H-----Ry[0]-----◯---
   #                          |   
   # q1: ---------------------CZ--

We then translate the circuit to a pure function :python:`param_to_st` that takes a set of parameters and an (optional) initial quantum state as its input.

.. code-block:: python

   param_to_st = qujax.get_params_to_statetensor_func(circuit_gates,
                                                      circuit_qubit_inds,
                                                      circuit_params_inds)
   
   param_to_st(jnp.array([0.1]))
   # Array([[0.58778524+0.j, 0.        +0.j],
   #        [0.80901706+0.j, 0.        +0.j]], dtype=complex64)

The optional initial state can be passed to :python:`param_to_st` using the :python:`statetensor_in` argument. When it is not provided, the initial state defaults to :math:`\ket{0...0}`.

Note that qujax represents quantum states as *statetensors*. For example, for :math:`N=4` qubits, the corresponding vector space has :math:`2^4` dimensions, and a uantum state in this space is represented by an array with shape :python:`(2,2,2,2)`. The usual statevector representation with shape :python:`(16,)` can be obtained by calling :python:`.flatten()` or :python:`.reshape(-1)` or :python:`.reshape(2**N)` on this array. 

In the statetensor representation, the coefficient associated with e.g. basis state :math:`\ket{0101}` is given by `arr[0,1,0,1]`; each axis corresponds to one qubit.

.. code-block:: python

   param_to_st(jnp.array([0.1])).flatten()
   # Array([0.58778524+0.j, 0.+0.j, 0.80901706+0.j, 0.+0.j], dtype=complex64)


Finally, by defining an observable, we can map the statetensor to an expectation value. A general observable is specified using lists of Pauli matrices, the qubits they act on, and the associated coefficients. 

For example, :math:`Z_1Z_2Z_3Z_4 - 2 X_3` would be written as :python:`[['Z','Z','Z','Z'], ['X']], [[1,2,3,4], [3]], [1., -2.]`.

.. code-block:: python

   st_to_expectation = qujax.get_statetensor_to_expectation_func([['Z']], [[0]], [1.])


Combining :python:`param_to_st` and :python:`st_to_expectation` gives us a parameter to expectation function that can be automatically differentiated using JAX.

.. code-block:: python

   from jax import value_and_grad

   param_to_expectation = lambda param: st_to_expectation(param_to_st(param))
   expectation_and_grad = value_and_grad(param_to_expectation)
   expectation_and_grad(jnp.array([0.1]))
   # (Array(-0.3090171, dtype=float32),
   #  Array([-2.987832], dtype=float32))

***********************
Mixed state simulation
***********************
Mixed state simulations are analogous to the above, but with calls to :doc:`densitytensor/get_params_to_densitytensor_func` and :doc:`densitytensor/get_densitytensor_to_expectation_func` instead.

.. code-block:: python
    
   param_to_dt = qujax.get_params_to_densitytensor_func(circuit_gates,
                                                        circuit_qubit_inds,
                                                        circuit_params_inds)
   dt = param_to_dt(jnp.array([0.1]))
   dt.shape
   # (2, 2, 2, 2)
   
   dt_to_expectation = qujax.get_densitytensor_to_expectation_func([['Z']], [[0]], [1.])
   dt_to_expectation(dt)
   # Array(-0.3090171, dtype=float32)

Similarly to a statetensor, which represents the reshaped :math:`2^N`-dimensional statevector of a pure quantum state, a *densitytensor* represents the reshaped :math:`2^N \times 2^N` density matrix of a mixed quantum state. This densitytensor has shape :python:`(2,) * 2 * N`.

For example, for :math:`N=2`, and a mixed state :math:`\frac{1}{2} (\ket{00}\bra{11} + \ket{11}\bra{00} + \ket{11}\bra{11} + \ket{00}\bra{00})`, the corresponding densitytensor :python:`dt` is such that :python:`dt[0,0,1,1] = dt[1,1,0,0] = dt[1,1,1,1] = dt[0,0,0,0] = 1/2`, and all other entries are zero.
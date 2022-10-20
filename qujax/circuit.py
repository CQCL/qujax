from __future__ import annotations
from typing import Sequence, Union, Callable, Protocol
from jax import numpy as jnp

from qujax import gates
from qujax.circuit_tools import check_circuit


class CallableArrayAndOptionalArray(Protocol):
    def __call__(self, params: jnp.ndarray, statetensor_in: jnp.ndarray = None) -> jnp.ndarray:
        ...


class CallableOptionalArray(Protocol):
    def __call__(self, statetensor_in: jnp.ndarray = None) -> jnp.ndarray:
        ...


UnionCallableOptionalArray = Union[CallableArrayAndOptionalArray, CallableOptionalArray]


def apply_gate(statetensor: jnp.ndarray, gate_unitary: jnp.ndarray, qubit_inds: Sequence[int]) -> jnp.ndarray:
    """
    Applies gate to statetensor and returns updated statetensor.
    Gate is represented by a unitary matrix (i.e. not parameterised).

    Args:
        statetensor: Input statetensor.
        gate_unitary: Unitary array representing gate
            must be in tensor form with shape (2,2,...).
        qubit_inds: Sequence of indices for gate to be applied to.
            2 * len(qubit_inds) is equal to the dimension of the gate unitary tensor.

    Returns:
        Updated statetensor.
    """
    statetensor = jnp.tensordot(gate_unitary, statetensor,
                                axes=(list(range(-len(qubit_inds), 0)), qubit_inds))
    statetensor = jnp.moveaxis(statetensor, list(range(len(qubit_inds))), qubit_inds)
    return statetensor


def _to_gate_funcs(gate_seq: Sequence[Union[str,
                                            jnp.ndarray,
                                            Callable[[jnp.ndarray], jnp.ndarray],
                                            Callable[[], jnp.ndarray]]])\
        -> Sequence[Callable[[jnp.ndarray], jnp.ndarray]]:
    """
    Ensures all gate_seq elements are functions that map (possibly empty) parameters
    to a unitary tensor.

    Args:
        gate_seq: Sequence of gates.
            Each element is either a string matching an array or function in qujax.gates,
            a unitary array (which will be reshaped into a tensor of shape (2,2,2,...) )
            or a function taking parameters and returning gate unitary in tensor form.

    Returns:
        Sequence of gate parameter to unitary functions

    """
    def _array_to_callable(arr: jnp.ndarray) -> Callable[[], jnp.ndarray]:
        return lambda: arr

    gate_seq_callable = []
    for gate in gate_seq:
        if isinstance(gate, str):
            gate = gates.__dict__[gate]

        if callable(gate):
            gate_func = gate
        elif hasattr(gate, '__array__'):
            gate_func = _array_to_callable(jnp.array(gate))
        else:
            raise TypeError(f'Unsupported gate type - gate must be either a string in qujax.gates, an array or '
                            f'callable: {gate}')
        gate_seq_callable.append(gate_func)

    return gate_seq_callable


def _arrayify_inds(param_inds_seq: Sequence[Sequence[int]]) -> Sequence[jnp.ndarray]:
    param_inds_seq = [jnp.array(p) for p in param_inds_seq]
    param_inds_seq = [jnp.array([]) if jnp.any(jnp.isnan(p)) else p.astype(int) for p in param_inds_seq]
    return param_inds_seq


def get_params_to_statetensor_func(gate_seq: Sequence[Union[str,
                                                            jnp.ndarray,
                                                            Callable[[jnp.ndarray], jnp.ndarray],
                                                            Callable[[], jnp.ndarray]]],
                                   qubit_inds_seq: Sequence[Sequence[int]],
                                   param_inds_seq: Sequence[Sequence[int]],
                                   n_qubits: int = None) -> UnionCallableOptionalArray:
    """
    Creates a function that maps circuit parameters to a statetensor.

    Args:
        gate_seq: Sequence of gates.
            Each element is either a string matching a unitary array or function in qujax.gates,
            a custom unitary array or a custom function taking parameters and returning a unitary array.
            Unitary arrays will be reshaped into tensor form (2, 2,...)
        qubit_inds_seq: Sequences of sequences representing qubit indices (ints) that gates are acting on.
            i.e. [[0], [0,1], [1]] tells qujax the first gate is a single qubit gate acting on the zeroth qubit,
            the second gate is a two qubit gate acting on the zeroth and first qubit etc.
        param_inds_seq: Sequence of sequences representing parameter indices that gates are using,
            i.e. [[0], [], [5, 2]] tells qujax that the first gate uses the zeroth parameter
            (the float at position zero in the parameter vector/array), the second gate is not parameterised
            and the third gates used the parameters at position five and two.
        n_qubits: Number of qubits, if fixed.

    Returns:
        Function which maps parameters (and optional statetensor_in) to a statetensor.
        If no parameters are found then the function only takes optional statetensor_in.

    """

    check_circuit(gate_seq, qubit_inds_seq, param_inds_seq, n_qubits)

    if n_qubits is None:
        n_qubits = max([max(qi) for qi in qubit_inds_seq]) + 1

    gate_seq_callable = _to_gate_funcs(gate_seq)
    param_inds_seq = _arrayify_inds(param_inds_seq)

    def params_to_statetensor_func(params: jnp.ndarray,
                                   statetensor_in: jnp.ndarray = None) -> jnp.ndarray:
        """
        Applies parameterised circuit (series of gates) to a statetensor_in (default is |0>^N).

        Args:
            params: Parameters of the circuit.
            statetensor_in: Optional. Input statetensor.
                Defaults to |0>^N (tensor of size 2^n with all zeroes except one in [0]*N index).

        Returns:
            Updated statetensor.

        """
        if statetensor_in is None:
            statetensor = jnp.zeros((2,) * n_qubits)
            statetensor = statetensor.at[(0,) * n_qubits].set(1.)
        else:
            statetensor = statetensor_in
        params = jnp.atleast_1d(params)
        for gate_func, qubit_inds, param_inds in zip(gate_seq_callable, qubit_inds_seq, param_inds_seq):
            gate_params = jnp.take(params, param_inds)
            gate_unitary = gate_func(*gate_params)
            gate_unitary = gate_unitary.reshape((2,) * (2 * len(qubit_inds)))   # Ensure gate is in tensor form
            statetensor = apply_gate(statetensor, gate_unitary, qubit_inds)
        return statetensor

    if all([pi.size == 0 for pi in param_inds_seq]):
        def no_params_to_statetensor_func(statetensor_in: jnp.ndarray = None) -> jnp.ndarray:
            """
            Applies circuit (series of gates with no parameters) to a statetensor_in (default is |0>^N).

            Args:
                statetensor_in: Optional. Input statetensor.
                    Defaults to |0>^N (tensor of size 2^n with all zeroes except one in [0]*N index).

            Returns:
                Updated statetensor.

            """
            return params_to_statetensor_func(jnp.array([]), statetensor_in)

        return no_params_to_statetensor_func

    return params_to_statetensor_func

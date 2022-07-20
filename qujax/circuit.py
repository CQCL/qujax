from typing import Sequence, Protocol, Union, Callable

from jax import numpy as jnp, random

from qujax import gates


class CallableOptionalArrayArg(Protocol):
    def __call__(self, x: Union[None, jnp.ndarray], y: jnp.ndarray = ..., /) -> jnp.ndarray:
        ...


def _get_apply_gate(gate_func: Callable,
                    qubit_inds: Sequence[int]):
    """
    Creates a function that applies a given gate_func to given qubit_inds of a statetensor.

    Args:
        gate_func: Function that takes any gate parameters and returns the gate unitary (in tensor form).
        qubit_inds: Sequence of indices for gate to be applied to.
            len(qubit_inds) is equal to the dimension of the gate unitary tensor.

    Returns:
        Function that takes statetensor and gate parameters, returns updated statetensor.

    """

    def apply_gate(statetensor: jnp.ndarray, params: jnp.ndarray) -> jnp.ndarray:
        """
        Applies {} gate to statetensor.

        Args:
            statetensor: Input statetensor.
            params: Gate parameters if any.

        Returns:
            Updated statetensor.
        """
        gate_unitary = gate_func(*params)
        statetensor = jnp.tensordot(gate_unitary, statetensor,
                                    axes=(list(range(-len(qubit_inds), 0)), qubit_inds))
        statetensor = jnp.moveaxis(statetensor, list(range(len(qubit_inds))), qubit_inds)
        return statetensor

    apply_gate.__doc__ = apply_gate.__doc__.format(gate_func.__name__)

    return apply_gate


def get_params_to_statetensor_func(gate_seq: Sequence[Union[str, Callable[[Union[None, jnp.ndarray]], jnp.ndarray]]],
                                   qubit_inds_seq: Sequence[Sequence[int]],
                                   param_inds_seq: Sequence[Sequence[int]],
                                   n_qubits: int = None) -> CallableOptionalArrayArg:
    """
    Creates a function that maps circuit parameters to a statetensor.

    Args:
        gate_seq: Sequence of gates.
            Each element is either a string matching a function in qujax.gates
            or a function taking parameters (can be empty) and returning gate unitary in tensor form.
        qubit_inds_seq: Sequences of qubits (ints) that gates are acting on.
        param_inds_seq: Sequence of parameter indices that gates are using, ie gate 3 uses 1st and 666th parameter.
        n_qubits: Number of qubits, if fixed.

    Returns:
        Function which maps parameters (and optional statetensor_in) to a statetensor.

    """
    if n_qubits is None:
        n_qubits = max([max(qi) for qi in qubit_inds_seq]) + 1

    gate_func_seq = [gates.__dict__[gate] if isinstance(gate, str) else gate for gate in gate_seq]
    apply_gate_seq = [_get_apply_gate(g, q) for g, q in zip(gate_func_seq, qubit_inds_seq) if g != 'Barrier']
    param_inds_seq = [jnp.array(p).astype(int) if p != [None] else jnp.array([]) for p in param_inds_seq]

    def apply_circuit(params: Union[None, jnp.ndarray],
                      statetensor_in: jnp.ndarray = None) -> jnp.ndarray:
        """
        Applies circuit (series of gates) to an statetensor_in (default is |0>^N).

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
        for gate_ind, apply_gate in enumerate(apply_gate_seq):
            gate_params = jnp.take(params, param_inds_seq[gate_ind])
            statetensor = apply_gate(statetensor, gate_params)
        return statetensor

    return apply_circuit


def integers_to_bitstrings(integers: Union[int, jnp.ndarray],
                           nbits: int = None) -> jnp.ndarray:
    """
    Convert integer or array of integers into their binary expansion(s).

    Args:
        integers: Integer or array of integer to be converted.
        nbits: Length of output binary expansion.

    Returns:
        Array of binary expansion(s).
    """
    integers = jnp.atleast_1d(integers)
    if nbits is None:
        nbits = (jnp.ceil(jnp.log2(integers.max()) + 1e-5)).astype(int)

    return jnp.squeeze(((integers[:, None] & (1 << jnp.arange(nbits - 1, -1, -1))) > 0).astype(int))


def bitstrings_to_integers(bitstrings: jnp.ndarray) -> Union[int, jnp.ndarray]:
    """
    Convert binary expansion(s) into integers.

    Args:
        bitstrings: Array of bitstring arrays.

    Returns:
        Array of integers.
    """
    bitstrings = jnp.atleast_2d(bitstrings)
    convarr = 2 ** jnp.arange(bitstrings.shape[-1] - 1, -1, -1)
    return jnp.squeeze(bitstrings.dot(convarr))


def sample_integers(random_key: random.PRNGKeyArray,
                    statetensor: jnp.ndarray,
                    n_samps: int = 1) -> jnp.ndarray:
    """
    Generate random integer samples according to statetensor.

    Args:
        random_key: JAX random key to seed samples.
        statetensor: Statetensor encoding sampling probabilities (in the form of amplitudes).
        n_samps: Number of samples to generate. Defaults to 1.

    Returns:
        Array with sampled integers, shape=(n_samps,).

    """
    sv_probs = jnp.square(jnp.abs(statetensor.flatten()))
    sampled_inds = random.choice(random_key, a=jnp.arange(statetensor.size), shape=(n_samps,), p=sv_probs)
    return sampled_inds


def sample_bitstrings(random_key: random.PRNGKeyArray,
                      statetensor: jnp.ndarray,
                      n_samps: int = 1) -> jnp.ndarray:
    """
    Generate random bitstring samples according to statetensor.

    Args:
        random_key: JAX random key to seed samples.
        statetensor: Statetensor encoding sampling probabilities (in the form of amplitudes).
        n_samps: Number of samples to generate. Defaults to 1.

    Returns:
        Array with sampled bitstrings, shape=(n_samps, statetensor.ndim).

    """
    return integers_to_bitstrings(sample_integers(random_key, statetensor, n_samps), statetensor.ndim)

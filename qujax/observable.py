from __future__ import annotations
from typing import Sequence, Callable, Union, Optional

from jax import numpy as jnp, random
from jax.lax import fori_loop

from qujax import gates


def _statetensor_to_single_expectation_func(gate_tensor: jnp.ndarray,
                                            qubit_inds: Sequence[int]) -> Callable[[jnp.ndarray], float]:
    """
    Creates a function that maps statetensor to its expected value under the given gate unitary and qubit indices.

    Args:
        gate_tensor: Gate unitary in tensor form.
        qubit_inds: Sequence of integer qubit indices to apply gate to.

    Returns:
        Function that takes statetensor and returns expected value (float).
    """

    def statetensor_to_single_expectation(statetensor: jnp.ndarray) -> float:
        """
        Evaluates expected value of statetensor through gate.

        Args:
            statetensor: Input statetensor.

        Returns:
            Expected value (float).
        """
        statetensor_new = jnp.tensordot(gate_tensor, statetensor,
                                        axes=(list(range(-len(qubit_inds), 0)), qubit_inds))
        statetensor_new = jnp.moveaxis(statetensor_new, list(range(len(qubit_inds))), qubit_inds)
        axes = tuple(range(statetensor.ndim))
        return jnp.tensordot(statetensor.conjugate(), statetensor_new, axes=(axes, axes)).real

    return statetensor_to_single_expectation


def get_statetensor_to_expectation_func(gate_seq_seq: Sequence[Sequence[Union[str, jnp.ndarray]]],
                                        qubits_seq_seq: Sequence[Sequence[int]],
                                        coefficients: Union[Sequence[float], jnp.ndarray]) \
        -> Callable[[jnp.ndarray], float]:
    """
    Converts gate strings, qubit indices and coefficients into a function that converts statetensor into expected value.

    Args:
        gate_seq_seq: Sequence of sequences of gates.
            Each gate is either a tensor (jnp.ndarray) or a string corresponding to a array in qujax.gates.
            E.g. [['Z', 'Z'], ['X']]
        qubits_seq_seq: Sequence of sequences of integer qubit indices.
            E.g. [[0,1], [2]]
        coefficients: Sequence of float coefficients to scale the expected values.

    Returns:
        Function that takes statetensor and returns expected value (float).
    """

    def get_gate_tensor(gate_seq: Sequence[Union[str, jnp.ndarray]]) -> jnp.ndarray:
        """
        Convert sequence of gate strings into single gate unitary (in tensor form).

        Args:
            gate_seq: Sequence of gate strings or arrays.

        Returns:
            Single gate unitary in tensor form (array).

        """
        single_gate_arrs = [gates.__dict__[gate] if isinstance(gate, str) else gate for gate in gate_seq]
        single_gate_arrs = [gate_arr.reshape((2,) * int(jnp.log2(gate_arr.size)))
                            for gate_arr in single_gate_arrs]
        full_gate_mat = single_gate_arrs[0]
        for single_gate_matrix in single_gate_arrs[1:]:
            full_gate_mat = jnp.kron(full_gate_mat, single_gate_matrix)
        full_gate_mat = full_gate_mat.reshape((2,) * int(jnp.log2(full_gate_mat.size)))
        return full_gate_mat

    apply_gate_funcs = [_statetensor_to_single_expectation_func(get_gate_tensor(gns), qi)
                        for gns, qi in zip(gate_seq_seq, qubits_seq_seq)]

    def statetensor_to_expectation_func(statetensor: jnp.ndarray) -> float:
        """
        Maps statetensor to expected value.

        Args:
            statetensor: Input statetensor.

        Returns:
            Expected value (float).

        """
        out = 0
        for coeff, f in zip(coefficients, apply_gate_funcs):
            out += coeff * f(statetensor)
        return out

    return statetensor_to_expectation_func


def get_statetensor_to_sampled_expectation_func(gate_seq_seq: Sequence[Sequence[Union[str, jnp.ndarray]]],
                                                qubits_seq_seq: Sequence[Sequence[int]],
                                                coefficients: Union[Sequence[float], jnp.ndarray]) \
        -> Callable[[jnp.ndarray, random.PRNGKeyArray, int], float]:
    """
    Converts gate strings, qubit indices and coefficients into a function that converts statetensor into
    a sampled expectation value.

    Args:
        gate_seq_seq: Sequence of sequences of gates.
            Each gate is either a tensor (jnp.ndarray) or a string corresponding to a array in qujax.gates.
            E.g. [['Z', 'Z'], ['X']]
        qubits_seq_seq: Sequence of sequences of integer qubit indices.
            E.g. [[0,1], [2]]
        coefficients: Sequence of float coefficients to scale the expected values.

    Returns:
        Function that takes statetensor, random key and integer number of shots
        and returns sampled expected value (float).
    """
    statetensor_to_expectation_func = get_statetensor_to_expectation_func(gate_seq_seq, qubits_seq_seq, coefficients)

    def statetensor_to_sampled_expectation_func(statetensor: jnp.ndarray,
                                                random_key: random.PRNGKeyArray,
                                                n_samps: int) -> float:
        """
        Maps statetensor to sampled expected value.

        Args:
            statetensor: Input statetensor.
            random_key: JAX random key
            n_samps: Number of samples contributing to sampled expectation.

        Returns:
            Sampled expected value (float).

        """
        sampled_integers = sample_integers(random_key, statetensor, n_samps)
        sampled_probs = fori_loop(0, n_samps,
                                  lambda i, sv: sv.at[sampled_integers[i]].add(1),
                                  jnp.zeros(statetensor.size))

        sampled_probs /= n_samps
        sampled_st = jnp.sqrt(sampled_probs).reshape(statetensor.shape)
        return statetensor_to_expectation_func(sampled_st)

    return statetensor_to_sampled_expectation_func


def integers_to_bitstrings(integers: Union[int, jnp.ndarray],
                           nbits: int = None) -> jnp.ndarray:
    """
    Convert integer or array of integers into their binary expansion(s).

    Args:
        integers: Integer or array of integers to be converted.
        nbits: Length of output binary expansion.
            Defaults to smallest possible.

    Returns:
        Array of binary expansion(s).
    """
    integers = jnp.atleast_1d(integers)
    if nbits is None:
        nbits = (jnp.ceil(jnp.log2(jnp.maximum(integers.max(), 1)) + 1e-5)).astype(int)

    return jnp.squeeze(((integers[:, None] & (1 << jnp.arange(nbits - 1, -1, -1))) > 0).astype(int))


def bitstrings_to_integers(bitstrings: jnp.ndarray) -> Union[int, jnp.ndarray]:
    """
    Convert binary expansion(s) into integers.

    Args:
        bitstrings: Bitstring array or array of bitstring arrays.

    Returns:
        Array of integers.
    """
    bitstrings = jnp.atleast_2d(bitstrings)
    convarr = 2 ** jnp.arange(bitstrings.shape[-1] - 1, -1, -1)
    return jnp.squeeze(bitstrings.dot(convarr)).astype(int)


def sample_integers(random_key: random.PRNGKeyArray,
                    statetensor: jnp.ndarray,
                    n_samps: Optional[int] = 1) -> jnp.ndarray:
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
                      n_samps: Optional[int] = 1) -> jnp.ndarray:
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

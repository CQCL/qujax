from __future__ import annotations
from typing import Sequence, Callable, Union, Optional

from jax import numpy as jnp, random
from jax.lax import fori_loop

from qujax.circuit import apply_gate
from qujax.gates import I, X, Y, Z

paulis = {'I': I, 'X': X, 'Y': Y, 'Z': Z}


def statetensor_to_single_expectation(statetensor: jnp.ndarray,
                                      hermitian: jnp.ndarray,
                                      qubit_inds: Sequence[int]) -> float:
    """
    Evaluates expected value of statetensor through a Hermitian matrix (in tensor form).

    Args:
        statetensor: Input statetensor.
        hermitian: Hermitian array
            must be in tensor form with shape (2,2,...).
        qubit_inds: Sequence of qubit indices for Hermitian to be applied to.
            Must have 2 * len(qubit_inds) = hermitian.ndim

    Returns:
        Expected value (float).
    """
    statetensor_new = apply_gate(statetensor, hermitian, qubit_inds)
    axes = tuple(range(statetensor.ndim))
    return jnp.tensordot(statetensor.conjugate(), statetensor_new, axes=(axes, axes)).real


def check_hermitian(hermitian: Union[str, jnp.ndarray]):
    """
    Checks whether a matrix or tensor is Hermitian.

    Args:
        hermitian: array containing potentially Hermitian matrix or tensor

    """
    if isinstance(hermitian, str):
        if hermitian not in paulis:
            raise TypeError(f'qujax only accepts {tuple(paulis.keys())} as Hermitian strings, received: {hermitian}')
    else:
        n_qubits = hermitian.ndim // 2
        hermitian_mat = hermitian.reshape(2 * n_qubits, 2 * n_qubits)
        if not jnp.allclose(hermitian_mat, hermitian_mat.T.conj()):
            raise TypeError(f'Array not Hermitian: {hermitian}')


def get_statetensor_to_expectation_func(hermitian_seq_seq: Sequence[Sequence[Union[str, jnp.ndarray]]],
                                        qubits_seq_seq: Sequence[Sequence[int]],
                                        coefficients: Union[Sequence[float], jnp.ndarray]) \
        -> Callable[[jnp.ndarray], float]:
    """
    Converts strings (or arrays) representing Hermitian matrices, qubit indices and
    coefficients into a function that converts a statetensor into an expected value.

    Args:
        hermitian_seq_seq: Sequence of sequences of Hermitian matrices/tensors.
            Each Hermitian is either a tensor (jnp.ndarray) or a string in ('X', 'Y', 'Z').
            E.g. [['Z', 'Z'], ['X']]
        qubits_seq_seq: Sequence of sequences of integer qubit indices.
            E.g. [[0,1], [2]]
        coefficients: Sequence of float coefficients to scale the expected values.

    Returns:
        Function that takes statetensor and returns expected value (float).
    """

    def get_hermitian_tensor(hermitian_seq: Sequence[Union[str, jnp.ndarray]]) -> jnp.ndarray:
        """
        Convert sequence of Hermitian strings/arrays into single array (in tensor form).

        Args:
            hermitian_seq: Sequence of Hermitian strings or arrays.

        Returns:
            Hermitian matrix in tensor form (array).

        """
        for h in hermitian_seq:
            check_hermitian(h)

        single_arrs = [paulis[h] if isinstance(h, str) else h for h in hermitian_seq]
        single_arrs = [h_arr.reshape((2,) * int(jnp.log2(h_arr.size))) for h_arr in single_arrs]


        full_mat = single_arrs[0]
        for single_matrix in single_arrs[1:]:
            full_mat = jnp.kron(full_mat, single_matrix)
        full_mat = full_mat.reshape((2,) * int(jnp.log2(full_mat.size)))
        return full_mat

    hermitian_tensors = [get_hermitian_tensor(h_seq) for h_seq in hermitian_seq_seq]

    def statetensor_to_expectation_func(statetensor: jnp.ndarray) -> float:
        """
        Maps statetensor to expected value.

        Args:
            statetensor: Input statetensor.

        Returns:
            Expected value (float).

        """
        out = 0
        for hermitian, qubit_inds, coeff in zip(hermitian_tensors, qubits_seq_seq, coefficients):
            out += coeff * statetensor_to_single_expectation(statetensor, hermitian, qubit_inds)
        return out

    return statetensor_to_expectation_func


def get_statetensor_to_sampled_expectation_func(hermitian_seq_seq: Sequence[Sequence[Union[str, jnp.ndarray]]],
                                                qubits_seq_seq: Sequence[Sequence[int]],
                                                coefficients: Union[Sequence[float], jnp.ndarray]) \
        -> Callable[[jnp.ndarray, random.PRNGKeyArray, int], float]:
    """
    Converts strings (or arrays) representing Hermitian matrices, qubit indices and
    coefficients into a function that converts a statetensor into a sampled expected value.

    Args:
        hermitian_seq_seq: Sequence of sequences of Hermitian matrices/tensors.
            Each Hermitian is either a tensor (jnp.ndarray) or a string in ('X', 'Y', 'Z').
            E.g. [['Z', 'Z'], ['X']]
        qubits_seq_seq: Sequence of sequences of integer qubit indices.
            E.g. [[0,1], [2]]
        coefficients: Sequence of float coefficients to scale the expected values.

    Returns:
        Function that takes statetensor, random key and integer number of shots
        and returns sampled expected value (float).
    """
    statetensor_to_expectation_func = get_statetensor_to_expectation_func(hermitian_seq_seq,
                                                                          qubits_seq_seq,
                                                                          coefficients)

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

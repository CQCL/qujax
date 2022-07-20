from typing import Tuple, Sequence

from jax import numpy as jnp
import pytket

from qujax.circuit import CallableOptionalArrayArg, get_params_to_statetensor_func


def _tk_qubits_to_inds(tk_qubits: Sequence[pytket.Qubit]) -> Tuple[int, ...]:
    """
    Convert Sequence of tket qubits objects to Tuple of integers qubit indices.

    Args:
        tk_qubits: Sequence of tket qubit object (as stored in pytket.Circuit.qubits).

    Returns:
        Tuple of qubit indices.
    """
    return tuple(q.index[0] for q in tk_qubits)


def tk_to_qujax(circuit: pytket.Circuit) -> CallableOptionalArrayArg:
    """
    Converts a tket circuit into a function that maps circuit parameters to a statetensor.
    Assumes all circuit gates can be found in qujax.gates.
    Input parameter to created function will be ordered as in circuit.get_commands()
        (pytket automatically reorders some gates, consider using Barriers).

    Args:
        circuit: pytket.Circuit object.

    Returns:
        Function which maps parameters (and optional statetensor_in) to a statetensor.

    """
    gate_name_seq = []
    qubit_inds_seq = []
    param_inds_seq = []
    param_index = 0
    for c in circuit.get_commands():
        gate_name = c.op.type.name
        if gate_name == 'Barrier':
            continue
        gate_name_seq.append(gate_name)
        qubit_inds_seq.append(_tk_qubits_to_inds(c.qubits))
        n_params = len(c.op.params)
        param_inds_seq.append(jnp.arange(param_index, param_index + n_params))
        param_index += n_params

    return get_params_to_statetensor_func(gate_name_seq, qubit_inds_seq, param_inds_seq, circuit.n_qubits)


def tk_to_qujax_symbolic(circuit: pytket.Circuit,
                        symbol_map: dict = None) -> CallableOptionalArrayArg:
    """
    Converts a tket circuit with symbolics parameters and a symbolic parameter map
        into a function that maps circuit parameters to a statetensor.
    Assumes all circuit gates can be found in qujax.gates.
    Note that the behaviour of tk_to_jax_symbolic(circuit) is different to tk_to_jax(circuit),
        tk_to_jax_symbolic will look for parameters in circuit.free_symbols() and if there are none
        it will assume that none of the gates require parameters.
        tk_to_jax will work out which gates are parameterised based on e.g. circuit.get_commands()[0].op.params.

    Args:
        circuit: pytket.Circuit object.
        symbol_map: dict that maps elements of circuit.free_symbols() (sympy) to parameter indices.

    Returns:
        Function which maps parameters (and optional statetensor_in) to a statetensor.

    """
    if symbol_map is None:
        free_symbols = circuit.free_symbols()
        n_symbols = len(free_symbols)
        symbol_map = dict(zip(free_symbols, range(n_symbols)))
    else:
        assert set(symbol_map.keys()) == circuit.free_symbols(), "Circuit keys do not much symbol_map"
        assert set(symbol_map.values()) == set(range(len(circuit.free_symbols()))), "Incorrect indices in symbol_map"

    gate_name_seq = []
    qubit_inds_seq = []
    param_inds_seq = []
    for c in circuit.get_commands():
        gate_name = c.op.type.name
        if gate_name == 'Barrier':
            continue
        gate_name_seq.append(gate_name)
        qubit_inds_seq.append(_tk_qubits_to_inds(c.qubits))
        param_inds_seq.append(jnp.array([symbol_map[symbol] for symbol in c.op.free_symbols()]))

    return get_params_to_statetensor_func(gate_name_seq, qubit_inds_seq, param_inds_seq, circuit.n_qubits)

import pytest
from sympy import Symbol
from jax import numpy as jnp, jit, grad, random

from pytket.circuit import Circuit

import quax


def _test_circuit(circuit, symbols):
    params = random.uniform(random.PRNGKey(0), (len(symbols),)) * 2
    param_map = dict(zip(symbols, params))
    symbol_map = dict(zip(symbols, range(len(symbols))))

    circuit_inst = circuit.copy()
    circuit_inst.symbol_substitution(param_map)
    true_sv = circuit_inst.get_statevector()

    apply_circuit = quax.tket.tk_to_quax_symbolic(circuit, symbol_map)

    test_sv = apply_circuit(params).flatten()
    assert jnp.all(jnp.abs(test_sv - true_sv) < 1e-5)

    jit_apply_circuit = jit(apply_circuit)
    test_jit_sv = jit_apply_circuit(params).flatten()
    assert jnp.all(jnp.abs(test_jit_sv - true_sv) < 1e-5)

    if len(params):
        cost_func = lambda p: jnp.square(apply_circuit(p)).real.sum()
        grad_cost_func = grad(cost_func)
        assert isinstance(grad_cost_func(params), jnp.ndarray)

        cost_jit_func = lambda p: jnp.square(jit_apply_circuit(p)).real.sum()
        grad_cost_jit_func = grad(cost_jit_func)
        assert isinstance(grad_cost_jit_func(params), jnp.ndarray)


def test_H():
    symbols = []

    circuit = Circuit(3)
    circuit.H(0)

    _test_circuit(circuit, symbols)


def test_CX():
    symbols = [Symbol("p0")]

    circuit = Circuit(2)
    circuit.H(0)
    circuit.Rz(symbols[0], 0)
    circuit.CX(0, 1)

    _test_circuit(circuit, symbols)


def test_CX_qrev():
    symbols = [Symbol("p0"), Symbol("p1")]

    circuit = Circuit(2)
    circuit.Rx(symbols[0], 0)
    circuit.Rx(symbols[1], 1)
    circuit.CX(1, 0)

    _test_circuit(circuit, symbols)


def test_CZ():
    symbols = [Symbol("p0")]

    circuit = Circuit(2)
    circuit.H(0)
    circuit.Rz(symbols[0], 0)
    circuit.CZ(0, 1)

    _test_circuit(circuit, symbols)


def test_CZ_qrev():
    symbols = [Symbol("p0")]

    circuit = Circuit(2)
    circuit.H(0)
    circuit.Rz(symbols[0], 0)
    circuit.CZ(1, 0)

    _test_circuit(circuit, symbols)


def test_CX_Barrier_Rx():
    symbols = [Symbol("p0"), Symbol("p1")]

    circuit = Circuit(3)
    circuit.CX(0, 1)
    circuit.add_barrier([0, 2])
    circuit.Rx(symbols[0], 0)
    circuit.Rx(symbols[1], 2)

    _test_circuit(circuit, symbols)


def test_circuit1():
    n_qubits = 4
    depth = 1
    symbols = [Symbol(f"p{j}") for j in range(n_qubits * (depth + 1))]

    circuit = Circuit(n_qubits)
    k = 0
    for i in range(n_qubits):
        circuit.Ry(symbols[k], i)
        k += 1
    for _ in range(depth):
        for i in range(0, n_qubits - 1, 2):
            circuit.CX(i, i + 1)
        for i in range(1, n_qubits - 1, 2):
            circuit.CX(i, i + 1)
        circuit.add_barrier(range(0, n_qubits))
        for i in range(n_qubits):
            circuit.Ry(symbols[k], i)
            k += 1

    _test_circuit(circuit, symbols)


def test_circuit2():
    n_qubits = 3
    depth = 1
    symbols = [Symbol(f"p{j}") for j in range(2 * n_qubits * (depth + 1))]

    circuit = Circuit(n_qubits)
    k = 0
    for i in range(n_qubits):
        circuit.H(i)
    for i in range(n_qubits):
        circuit.Rz(symbols[k], i)
        k += 1
    for i in range(n_qubits):
        circuit.Rx(symbols[k], i)
        k += 1
    for _ in range(depth):
        for i in range(0, n_qubits - 1):
            circuit.CZ(i, i + 1)
        circuit.add_barrier(range(0, n_qubits))
        for i in range(n_qubits):
            circuit.Rz(symbols[k], i)
            k += 1
        for i in range(n_qubits):
            circuit.Rx(symbols[k], i)
            k += 1

    _test_circuit(circuit, symbols)


def test_HH():
    circuit = Circuit(3)
    circuit.H(0)

    apply_circuit = quax.tket.tk_to_quax_symbolic(circuit)

    st1 = apply_circuit(None)
    st2 = apply_circuit(None, st1)
    all_zeros_sv = jnp.array(jnp.arange(st2.size) == 0, dtype=int)
    assert jnp.all(jnp.abs(st2.flatten() - all_zeros_sv) < 1e-5)


def test_exception_symbol_map():
    symbols = [Symbol("p0"), Symbol("p1"), Symbol("bad_bad_symbol")]

    circuit = Circuit(2)
    circuit.Rx(symbols[0], 0)
    circuit.Rx(symbols[1], 1)
    circuit.CX(1, 0)

    with pytest.raises(AssertionError):
        _test_circuit(circuit, symbols)

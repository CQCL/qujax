from jax import numpy as jnp, jit, grad, random

import qujax

from pytket.circuit import Circuit, Qubit
from pytket.pauli import Pauli, QubitPauliString
from pytket.utils import QubitPauliOperator


def _test_circuit(circuit, param):
    true_sv = circuit.get_statevector()

    apply_circuit = qujax.tket.tk_to_qujax(circuit)

    test_st = apply_circuit(param)
    test_sv = test_st.flatten()
    assert jnp.all(jnp.abs(test_sv - true_sv) < 1e-5)

    jit_apply_circuit = jit(apply_circuit)
    test_jit_sv = jit_apply_circuit(param).flatten()
    assert jnp.all(jnp.abs(test_jit_sv - true_sv) < 1e-5)

    if param is not None:
        cost_func = lambda p: jnp.square(apply_circuit(p)).real.sum()
        grad_cost_func = grad(cost_func)
        assert isinstance(grad_cost_func(param), jnp.ndarray)

        cost_jit_func = lambda p: jnp.square(jit_apply_circuit(p)).real.sum()
        grad_cost_jit_func = grad(cost_jit_func)
        assert isinstance(grad_cost_jit_func(param), jnp.ndarray)


def test_H():
    circuit = Circuit(3)
    circuit.H(0)

    _test_circuit(circuit, None)


def test_CX():
    param = jnp.array([0.25])

    circuit = Circuit(2)
    circuit.H(0)
    circuit.Rz(param[0], 0)
    circuit.CX(0, 1)

    _test_circuit(circuit, param)


def test_CX_callable():
    param = jnp.array([0.25])

    def H():
        return qujax.gates._H

    def Rz(p):
        return qujax.gates.Rz(p)

    def CX():
        return qujax.gates._CX

    gate_seq = [H, Rz, CX]
    qubit_inds_seq = [[0], [0], [0, 1]]
    param_inds_seq = [[], [0], []]

    apply_circuit = qujax.get_params_to_statetensor_func(gate_seq,
                                                        qubit_inds_seq,
                                                        param_inds_seq)

    circuit = Circuit(2)
    circuit.H(0)
    circuit.Rz(param[0], 0)
    circuit.CX(0, 1)
    true_sv = circuit.get_statevector()

    test_st = apply_circuit(param)
    test_sv = test_st.flatten()
    assert jnp.all(jnp.abs(test_sv - true_sv) < 1e-5)

    jit_apply_circuit = jit(apply_circuit)
    test_jit_sv = jit_apply_circuit(param).flatten()
    assert jnp.all(jnp.abs(test_jit_sv - true_sv) < 1e-5)


def test_CX_qrev():
    param = jnp.array([0.2, 0.8])

    circuit = Circuit(2)
    circuit.Rx(param[0], 0)
    circuit.Rx(param[1], 1)
    circuit.CX(1, 0)

    _test_circuit(circuit, param)


def test_CZ():
    param = jnp.array([0.25])

    circuit = Circuit(2)
    circuit.H(0)
    circuit.Rz(param[0], 0)
    circuit.CZ(0, 1)

    _test_circuit(circuit, param)


def test_CZ_qrev():
    param = jnp.array([0.25])

    circuit = Circuit(2)
    circuit.H(0)
    circuit.Rz(param[0], 0)
    circuit.CZ(1, 0)

    _test_circuit(circuit, param)


def test_CX_Barrier_Rx():
    param = jnp.array([0, 1 / jnp.pi])

    circuit = Circuit(3)
    circuit.CX(0, 1)
    circuit.add_barrier([0, 2])
    circuit.Rx(param[0], 0)
    circuit.Rx(param[1], 2)

    _test_circuit(circuit, param)


def test_circuit1():
    n_qubits = 4
    depth = 1

    param = random.uniform(random.PRNGKey(0), (n_qubits * (depth + 1),)) * 2

    circuit = Circuit(n_qubits)

    k = 0
    for i in range(n_qubits):
        circuit.Ry(param[k], i)
        k += 1

    for _ in range(depth):
        for i in range(0, n_qubits - 1, 2):
            circuit.CX(i, i + 1)
        for i in range(1, n_qubits - 1, 2):
            circuit.CX(i, i + 1)
        circuit.add_barrier(range(0, n_qubits))
        for i in range(n_qubits):
            circuit.Ry(param[k], i)
            k += 1

    _test_circuit(circuit, param)


def test_circuit2():
    n_qubits = 3
    depth = 1

    param = random.uniform(random.PRNGKey(0), (2 * n_qubits * (depth + 1),)) * 2

    circuit = Circuit(n_qubits)

    k = 0
    for i in range(n_qubits):
        circuit.H(i)
    for i in range(n_qubits):
        circuit.Rz(param[k], i)
        k += 1
    for i in range(n_qubits):
        circuit.Rx(param[k], i)
        k += 1

    for _ in range(depth):
        for i in range(0, n_qubits - 1):
            circuit.CZ(i, i + 1)
        circuit.add_barrier(range(0, n_qubits))
        for i in range(n_qubits):
            circuit.Rz(param[k], i)
            k += 1
        for i in range(n_qubits):
            circuit.Rx(param[k], i)
            k += 1

    _test_circuit(circuit, param)


def test_HH():
    circuit = Circuit(3)
    circuit.H(0)

    apply_circuit = qujax.tket.tk_to_qujax(circuit)

    st1 = apply_circuit(None)
    st2 = apply_circuit(None, st1)

    all_zeros_sv = jnp.array(jnp.arange(st2.size) == 0, dtype=int)

    assert jnp.all(jnp.abs(st2.flatten() - all_zeros_sv) < 1e-5)


def test_quantum_hamiltonian():
    n_qubits = 5

    strings_zz = [QubitPauliString({Qubit(j): Pauli.Z, Qubit(j + 1): Pauli.Z}) for j in range(n_qubits - 1)]
    coefs_zz = random.normal(random.PRNGKey(0), shape=(len(strings_zz),))
    tket_op_dict_zz = dict(zip(strings_zz, coefs_zz))
    strings_x = [QubitPauliString({Qubit(j): Pauli.X}) for j in range(n_qubits)]
    coefs_x = random.normal(random.PRNGKey(0), shape=(len(strings_x),))
    tket_op_dict_x = dict(zip(strings_x, coefs_x))
    tket_op = QubitPauliOperator(tket_op_dict_zz | tket_op_dict_x)

    gate_str_seq_seq = [['Z', 'Z']] * (n_qubits - 1) + [['X']] * n_qubits
    qubit_inds_seq = [[i, i + 1] for i in range(n_qubits - 1)] + [[i] for i in range(n_qubits)]
    st_to_exp = qujax.get_statetensor_to_expectation_func(gate_str_seq_seq,
                                                         qubit_inds_seq,
                                                         jnp.concatenate([coefs_zz, coefs_x]))

    state = random.uniform(random.PRNGKey(0), shape=(2 ** n_qubits,))
    state /= jnp.linalg.norm(state)

    tket_exp = tket_op.state_expectation(state)
    jax_exp = st_to_exp(state.reshape((2,) * n_qubits))

    assert jnp.abs(tket_exp - jax_exp) < 1e-5

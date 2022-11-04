from itertools import combinations
from jax import numpy as jnp, jit

import qujax
from qujax import get_params_to_statetensor_func
from qujax import _kraus_single, kraus, get_params_to_densitytensor_func, partial_trace
from qujax.observable import statetensor_to_densitytensor
from qujax import densitytensor_to_measurement_probabilities, densitytensor_to_measured_densitytensor


def test_kraus_single():
    n_qubits = 3
    dim = 2 ** n_qubits
    density_matrix = jnp.arange(dim ** 2).reshape(dim, dim)
    density_tensor = density_matrix.reshape((2,) * 2 * n_qubits)
    kraus_operator = qujax.gates.Rx(0.2)

    qubit_inds = (1,)

    unitary_matrix = jnp.kron(jnp.eye(2 * qubit_inds[0]), kraus_operator)
    unitary_matrix = jnp.kron(unitary_matrix, jnp.eye(2 * (n_qubits - qubit_inds[-1] - 1)))
    check_kraus_dm = unitary_matrix @ density_matrix @ unitary_matrix.conj().T

    # qujax._kraus_single
    qujax_kraus_dt = _kraus_single(density_tensor, kraus_operator, qubit_inds)
    qujax_kraus_dm = qujax_kraus_dt.reshape(dim, dim)

    assert jnp.allclose(qujax_kraus_dm, check_kraus_dm)

    qujax_kraus_dt_jit = jit(_kraus_single, static_argnums=(2,))(density_tensor, kraus_operator, qubit_inds)
    qujax_kraus_dm_jit = qujax_kraus_dt_jit.reshape(dim, dim)
    assert jnp.allclose(qujax_kraus_dm_jit, check_kraus_dm)

    # qujax.kraus (but for a single array)
    qujax_kraus_dt = kraus(density_tensor, kraus_operator, qubit_inds)
    qujax_kraus_dm = qujax_kraus_dt.reshape(dim, dim)
    assert jnp.allclose(qujax_kraus_dm, check_kraus_dm)

    qujax_kraus_dt_jit = jit(kraus, static_argnums=(2,))(density_tensor, kraus_operator, qubit_inds)
    qujax_kraus_dm_jit = qujax_kraus_dt_jit.reshape(dim, dim)
    assert jnp.allclose(qujax_kraus_dm_jit, check_kraus_dm)


def test_kraus_single_2qubit():
    n_qubits = 4
    dim = 2 ** n_qubits
    density_matrix = jnp.arange(dim ** 2).reshape(dim, dim)
    density_tensor = density_matrix.reshape((2,) * 2 * n_qubits)
    kraus_operator_tensor = qujax.gates.ZZPhase(0.1)
    kraus_operator = qujax.gates.ZZPhase(0.1).reshape(4, 4)

    qubit_inds = (1, 2)

    unitary_matrix = jnp.kron(jnp.eye(2 * qubit_inds[0]), kraus_operator)
    unitary_matrix = jnp.kron(unitary_matrix, jnp.eye(2 * (n_qubits - qubit_inds[-1] - 1)))
    check_kraus_dm = unitary_matrix @ density_matrix @ unitary_matrix.conj().T

    # qujax._kraus_single
    qujax_kraus_dt = _kraus_single(density_tensor, kraus_operator_tensor, qubit_inds)
    qujax_kraus_dm = qujax_kraus_dt.reshape(dim, dim)

    assert jnp.allclose(qujax_kraus_dm, check_kraus_dm)

    qujax_kraus_dt_jit = jit(_kraus_single, static_argnums=(2,))(density_tensor, kraus_operator_tensor, qubit_inds)
    qujax_kraus_dm_jit = qujax_kraus_dt_jit.reshape(dim, dim)
    assert jnp.allclose(qujax_kraus_dm_jit, check_kraus_dm)

    # qujax.kraus (but for a single array)
    qujax_kraus_dt = kraus(density_tensor, kraus_operator_tensor, qubit_inds)
    qujax_kraus_dm = qujax_kraus_dt.reshape(dim, dim)
    assert jnp.allclose(qujax_kraus_dm, check_kraus_dm)

    qujax_kraus_dt = kraus(density_tensor, kraus_operator, qubit_inds)  # check reshape kraus_operator correctly
    qujax_kraus_dm = qujax_kraus_dt.reshape(dim, dim)
    assert jnp.allclose(qujax_kraus_dm, check_kraus_dm)

    qujax_kraus_dt_jit = jit(kraus, static_argnums=(2,))(density_tensor, kraus_operator_tensor, qubit_inds)
    qujax_kraus_dm_jit = qujax_kraus_dt_jit.reshape(dim, dim)
    assert jnp.allclose(qujax_kraus_dm_jit, check_kraus_dm)


def test_kraus_multiple():
    n_qubits = 3
    dim = 2 ** n_qubits
    density_matrix = jnp.arange(dim ** 2).reshape(dim, dim)
    density_tensor = density_matrix.reshape((2,) * 2 * n_qubits)

    kraus_operators = [0.25 * qujax.gates.H, 0.25 * qujax.gates.Rx(0.3), 0.5 * qujax.gates.Ry(0.1)]

    qubit_inds = (1,)

    unitary_matrices = [jnp.kron(jnp.eye(2 * qubit_inds[0]), ko) for ko in kraus_operators]
    unitary_matrices = [jnp.kron(um, jnp.eye(2 * (n_qubits - qubit_inds[0] - 1))) for um in unitary_matrices]

    check_kraus_dm = jnp.zeros_like(density_matrix)
    for um in unitary_matrices:
        check_kraus_dm += um @ density_matrix @ um.conj().T

    qujax_kraus_dt = kraus(density_tensor, kraus_operators, qubit_inds)
    qujax_kraus_dm = qujax_kraus_dt.reshape(dim, dim)

    assert jnp.allclose(qujax_kraus_dm, check_kraus_dm)

    qujax_kraus_dt_jit = jit(kraus, static_argnums=(2,))(density_tensor, kraus_operators, qubit_inds)
    qujax_kraus_dm_jit = qujax_kraus_dt_jit.reshape(dim, dim)
    assert jnp.allclose(qujax_kraus_dm_jit, check_kraus_dm)


def test_params_to_densitytensor_func():
    n_qubits = 2

    gate_seq = ["Rx" for _ in range(n_qubits)]
    qubit_inds_seq = [(i,) for i in range(n_qubits)]
    param_inds_seq = [(i,) for i in range(n_qubits)]

    gate_seq += ["CZ" for _ in range(n_qubits - 1)]
    qubit_inds_seq += [(i, i + 1) for i in range(n_qubits - 1)]
    param_inds_seq += [() for _ in range(n_qubits - 1)]

    params_to_dt = get_params_to_densitytensor_func(gate_seq, qubit_inds_seq, param_inds_seq, n_qubits)
    params_to_st = get_params_to_statetensor_func(gate_seq, qubit_inds_seq, param_inds_seq, n_qubits)

    params = jnp.arange(n_qubits) / 10.

    st = params_to_st(params)
    dt_test = statetensor_to_densitytensor(st)

    dt = params_to_dt(params)

    assert jnp.allclose(dt, dt_test)

    jit_dt = jit(params_to_dt)(params)
    assert jnp.allclose(jit_dt, dt_test)


def test_params_to_densitytensor_func_with_bit_flip():
    n_qubits = 2

    gate_seq = ["Rx" for _ in range(n_qubits)]
    qubit_inds_seq = [(i,) for i in range(n_qubits)]
    param_inds_seq = [(i,) for i in range(n_qubits)]

    gate_seq += ["CZ" for _ in range(n_qubits - 1)]
    qubit_inds_seq += [(i, i + 1) for i in range(n_qubits - 1)]
    param_inds_seq += [() for _ in range(n_qubits - 1)]

    params_to_pre_bf_st = get_params_to_statetensor_func(gate_seq, qubit_inds_seq, param_inds_seq, n_qubits)

    kraus_ops = [[0.3 * jnp.eye(2), 0.7 * qujax.gates.X]]
    kraus_qubit_inds = [(0,)]
    kraus_param_inds = [None]

    gate_seq += kraus_ops
    qubit_inds_seq += kraus_qubit_inds
    param_inds_seq += kraus_param_inds

    params_to_dt = get_params_to_densitytensor_func(gate_seq, qubit_inds_seq, param_inds_seq, n_qubits)

    params = jnp.arange(n_qubits) / 10.

    pre_bf_st = params_to_pre_bf_st(params)
    pre_bf_dt = (pre_bf_st.reshape(-1, 1) @ pre_bf_st.reshape(1, -1).conj()).reshape(2 for _ in range(2 * n_qubits))
    dt_test = kraus(pre_bf_dt, kraus_ops[0], kraus_qubit_inds[0])

    dt = params_to_dt(params)

    assert jnp.allclose(dt, dt_test)

    jit_dt = jit(params_to_dt)(params)
    assert jnp.allclose(jit_dt, dt_test)


def test_partial_trace_1():
    state1 = 1 / jnp.sqrt(2) * jnp.array([1., 1.])
    state2 = jnp.kron(state1, state1)
    state3 = jnp.kron(state1, state2)

    dt1 = jnp.outer(state1, state1.conj()).reshape((2,) * 2)
    dt2 = jnp.outer(state2, state2.conj()).reshape((2,) * 4)
    dt3 = jnp.outer(state3, state3.conj()).reshape((2,) * 6)

    for i in range(3):
        assert jnp.allclose(partial_trace(dt3, [i]), dt2)

    for i in combinations(range(3), 2):
        assert jnp.allclose(partial_trace(dt3, i), dt1)


def test_partial_trace_2():
    n_qubits = 3

    gate_seq = ["Rx" for _ in range(n_qubits)]
    qubit_inds_seq = [(i,) for i in range(n_qubits)]
    param_inds_seq = [(i,) for i in range(n_qubits)]

    gate_seq += ["CZ" for _ in range(n_qubits - 1)]
    qubit_inds_seq += [(i, i + 1) for i in range(n_qubits - 1)]
    param_inds_seq += [() for _ in range(n_qubits - 1)]

    params_to_dt = get_params_to_densitytensor_func(gate_seq, qubit_inds_seq, param_inds_seq, n_qubits)

    params = jnp.arange(1, n_qubits + 1) / 10.

    dt = params_to_dt(params)
    dt_discard_test = jnp.trace(dt, axis1=0, axis2=n_qubits)
    dt_discard = partial_trace(dt, [0])

    assert jnp.allclose(dt_discard, dt_discard_test)


def test_measure():
    n_qubits = 3

    gate_seq = ["Rx" for _ in range(n_qubits)]
    qubit_inds_seq = [(i,) for i in range(n_qubits)]
    param_inds_seq = [(i,) for i in range(n_qubits)]

    gate_seq += ["CZ" for _ in range(n_qubits - 1)]
    qubit_inds_seq += [(i, i + 1) for i in range(n_qubits - 1)]
    param_inds_seq += [() for _ in range(n_qubits - 1)]

    params_to_dt = get_params_to_densitytensor_func(gate_seq, qubit_inds_seq, param_inds_seq, n_qubits)

    params = jnp.arange(1, n_qubits + 1) / 10.

    dt = params_to_dt(params)

    qubit_inds = [0]

    all_probs = jnp.diag(dt.reshape(2 ** n_qubits, 2 ** n_qubits)).real
    all_probs_marginalise \
        = all_probs.reshape((2,) * n_qubits).sum(axis=[i for i in range(n_qubits) if i not in qubit_inds])

    probs = densitytensor_to_measurement_probabilities(dt, qubit_inds)

    assert jnp.isclose(probs.sum(), 1.)
    assert jnp.isclose(all_probs.sum(), 1.)
    assert jnp.allclose(probs, all_probs_marginalise)

    dm = dt.reshape(2 ** n_qubits, 2 ** n_qubits)
    projector = jnp.array([[1, 0], [0, 0]])
    for _ in range(n_qubits - 1):
        projector = jnp.kron(projector, jnp.eye(2))
    measured_dm = projector @ dm @ projector.T.conj()
    measured_dm /= jnp.trace(projector.T.conj() @ projector @ dm)
    measured_dt_true = measured_dm.reshape((2,) * 2 * n_qubits)

    measured_dt = densitytensor_to_measured_densitytensor(dt, qubit_inds, 0)
    measured_dt_bits = densitytensor_to_measured_densitytensor(dt, qubit_inds, (0,)*n_qubits)
    assert jnp.allclose(measured_dt_true, measured_dt)
    assert jnp.allclose(measured_dt_true, measured_dt_bits)

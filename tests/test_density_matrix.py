from jax import numpy as jnp, jit

import qujax
from qujax.density_matrix import _kraus_single, get_params_to_densitytensor_func
from qujax import get_params_to_statetensor_func


def test_kraus_single():
    n_qubits = 3
    dim = 2 ** n_qubits
    density_matrix = jnp.arange(dim**2).reshape(dim, dim)
    density_tensor = density_matrix.reshape((2,) * 2 * n_qubits)
    kraus_operator = qujax.gates.Rx(0.2)

    qubit_inds = (1,)

    qujax_kraus_dt = _kraus_single(density_tensor, kraus_operator, qubit_inds)
    qujax_kraus_dm = qujax_kraus_dt.reshape(dim, dim)

    unitary_matrix = jnp.kron(jnp.eye(2 * qubit_inds[0]), kraus_operator)
    unitary_matrix = jnp.kron(unitary_matrix, jnp.eye(2 * (n_qubits - qubit_inds[0] - 1)))
    check_kraus_dm = unitary_matrix @ density_matrix @ unitary_matrix.conj().T

    assert jnp.all(jnp.abs(qujax_kraus_dm - check_kraus_dm) < 1e-5)

    qujax_kraus_dt_jit = jit(_kraus_single, static_argnums=(2,))(density_tensor, kraus_operator, qubit_inds)
    qujax_kraus_dm_jit = qujax_kraus_dt_jit.reshape(dim, dim)
    assert jnp.all(jnp.abs(qujax_kraus_dm_jit - check_kraus_dm) < 1e-5)


def test_params_to_densitytensor_func():
    n_qubits = 2

    gate_seq = ["Rx" for _ in range(n_qubits)]
    qubit_inds_seq = [(i,) for i in range(n_qubits)]
    param_inds_seq = [(i,) for i in range(n_qubits)]

    gate_seq += ["CZ" for _ in range(n_qubits - 1)]
    qubit_inds_seq += [(i, i+1) for i in range(n_qubits - 1)]
    param_inds_seq += [() for _ in range(n_qubits - 1)]

    params_to_dt = get_params_to_densitytensor_func(gate_seq, qubit_inds_seq, param_inds_seq, n_qubits)
    params_to_st = get_params_to_statetensor_func(gate_seq, qubit_inds_seq, param_inds_seq, n_qubits)

    params = jnp.arange(n_qubits)/10.

    st = params_to_st(params)
    dt_test = (st.reshape(-1, 1) @ st.reshape(1, -1).conj()).reshape(2 for _ in range(2*n_qubits))

    dt = params_to_dt(params)

    assert jnp.allclose(dt, dt_test)


def test_discards():
    n_qubits = 3

    gate_seq = ["Rx" for _ in range(n_qubits)]
    qubit_inds_seq = [(i,) for i in range(n_qubits)]
    param_inds_seq = [(i,) for i in range(n_qubits)]

    gate_seq += ["CZ" for _ in range(n_qubits - 1)]
    qubit_inds_seq += [(i, i+1) for i in range(n_qubits - 1)]
    param_inds_seq += [() for _ in range(n_qubits - 1)]

    params_to_dt = get_params_to_densitytensor_func(gate_seq, qubit_inds_seq, param_inds_seq, n_qubits)

    gate_seq.append("discard")
    qubits_to_discard = (1, 2,)
    qubit_inds_seq.append(qubits_to_discard)
    param_inds_seq.append(())

    params_to_dt_w_discard = get_params_to_densitytensor_func(gate_seq, qubit_inds_seq, param_inds_seq, n_qubits)

    params = jnp.arange(1, n_qubits+1)/10.

    dt = params_to_dt(params)
    
    dt_discard_test = dt
    for n, q in enumerate(reversed(qubits_to_discard)):
        dt_discard_test = jnp.trace(dt_discard_test, axis1=q, axis2=n_qubits + q - n)

    dt_discard = params_to_dt_w_discard(params)

    assert jnp.allclose(dt_discard, dt_discard_test)


def test_creation():
    statetensor = jnp.array([[1., 0.], [0., 0.]])
    gate_seq = ["create", "create", "create"]
    qubit_inds_seq = [(1,), (2,), (3,)]
    params_inds_seq = [(0,), (0,), (0,)]

    params = [statetensor]

    params_to_dt = get_params_to_densitytensor_func(gate_seq, qubit_inds_seq, params_inds_seq, 1)

    statetensor_test = jnp.zeros((2,) * 4 * 2)
    statetensor_test = statetensor_test.at[(0,) * 4 * 2].set(1.)

    dt = params_to_dt(params, statetensor)

    assert jnp.allclose(dt, statetensor_test)


def test_creating_and_discarding():
    statetensor = jnp.array([[1., 0.], [0., 0.]])
    gate_seq = ["create", "discard", "create", "discard"]
    qubit_inds_seq = [(2,), (0,), (3,), (2,)]
    params_inds_seq = [(0,), (), (0,), ()]

    params = [statetensor]

    params_to_dt = get_params_to_densitytensor_func(gate_seq, qubit_inds_seq, params_inds_seq, 1)

    dt = params_to_dt(params, statetensor)

    assert jnp.allclose(dt, statetensor)

from jax import numpy as jnp, jit

import qujax


def test_H():
    gates = ['H']
    qubits = [[0]]
    param_inds = [[]]

    param_to_st = qujax.get_params_to_statetensor_func(gates, qubits, param_inds)
    st = param_to_st()
    st_jit = jit(param_to_st)()

    true_sv = jnp.array([0.70710678 + 0.j, 0.70710678 + 0.j])

    assert st.size == true_sv.size
    assert jnp.allclose(unitary @ zero_sv, true_sv)
    assert jnp.allclose(unitary_jit @ zero_sv, true_sv)

    param_to_unitary = qujax.get_params_to_unitarytensor_func(gates, qubits, param_inds)
    unitary = param_to_unitary().reshape(2, 2)
    unitary_jit = jit(param_to_unitary)().reshape(2, 2)
    zero_sv = jnp.zeros(2).at[0].set(1)
    assert jnp.allclose(unitary @ zero_sv, true_sv)
    assert jnp.allclose(unitary_jit @ zero_sv, true_sv)


def test_H_redundant_qubits():
    gates = ['H']
    qubits = [[0]]
    param_inds = [[]]
    n_qubits = 3

    param_to_st = qujax.get_params_to_statetensor_func(gates, qubits, param_inds, n_qubits)
    st = param_to_st(statetensor_in=None)

    true_sv = jnp.array([0.70710678, 0., 0., 0.,
                         0.70710678, 0., 0., 0.])

    assert st.size == true_sv.size
    assert jnp.all(jnp.abs(st.flatten() - true_sv) < 1e-5)

    param_to_unitary = qujax.get_params_to_unitarytensor_func(gates, qubits, param_inds, n_qubits)
    unitary = param_to_unitary().reshape(2 ** n_qubits, 2 ** n_qubits)
    unitary_jit = jit(param_to_unitary)().reshape(2 ** n_qubits, 2 ** n_qubits)
    zero_sv = jnp.zeros(2 ** n_qubits).at[0].set(1)
    assert jnp.allclose(unitary @ zero_sv, true_sv)
    assert jnp.allclose(unitary_jit @ zero_sv, true_sv)


def test_CX_Rz_CY():
    gates = ['H', 'H', 'H', 'CX', 'Rz', 'CY']
    qubits = [[0], [1], [2], [0, 1], [1], [1, 2]]
    param_inds = [[], [], [], None, [0], []]

    param_to_st = qujax.get_params_to_statetensor_func(gates, qubits, param_inds)
    param = jnp.array(0.1)
    st = param_to_st(param)

    true_sv = jnp.array([0.34920055 - 0.05530793j, 0.34920055 - 0.05530793j,
                         0.05530793 - 0.34920055j, -0.05530793 + 0.34920055j,
                         0.34920055 - 0.05530793j, 0.34920055 - 0.05530793j,
                         0.05530793 - 0.34920055j, -0.05530793 + 0.34920055j], dtype='complex64')

    assert st.size == true_sv.size
    assert jnp.all(jnp.abs(st.flatten() - true_sv) < 1e-5)

    n_qubits = 3
    param_to_unitary = qujax.get_params_to_unitarytensor_func(gates, qubits, param_inds, n_qubits)
    unitary = param_to_unitary(param).reshape(2 ** n_qubits, 2 ** n_qubits)
    unitary_jit = jit(param_to_unitary)(param).reshape(2 ** n_qubits, 2 ** n_qubits)
    zero_sv = jnp.zeros(2 ** n_qubits).at[0].set(1)
    assert jnp.all(jnp.abs(unitary @ zero_sv - true_sv) < 1e-5)
    assert jnp.all(jnp.abs(unitary_jit @ zero_sv - true_sv) < 1e-5)


def test_stacked_circuits():
    gates = ['H']
    qubits = [[0]]
    param_inds = [[]]

    param_to_st = qujax.get_params_to_statetensor_func(gates, qubits, param_inds)

    st1 = param_to_st()
    st2 = param_to_st(st1)

    st2_2 = param_to_st(statetensor_in=st1)

    all_zeros_sv = jnp.array(jnp.arange(st2.size) == 0, dtype=int)

    assert jnp.all(jnp.abs(st2.flatten() - all_zeros_sv) < 1e-5)
    assert jnp.all(jnp.abs(st2_2.flatten() - all_zeros_sv) < 1e-5)


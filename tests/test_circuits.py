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
    assert jnp.all(jnp.abs(st.flatten() - true_sv) < 1e-5)
    assert jnp.all(jnp.abs(st_jit.flatten() - true_sv) < 1e-5)


def test_H_redundant_qubits():
    gates = ['H']
    qubits = [[0]]
    param_inds = [[]]

    param_to_st = qujax.get_params_to_statetensor_func(gates, qubits, param_inds, 3)
    st = param_to_st(statetensor_in=None)

    true_sv = jnp.array([0.70710678, 0., 0., 0.,
                         0.70710678, 0., 0., 0.])

    assert st.size == true_sv.size
    assert jnp.all(jnp.abs(st.flatten() - true_sv) < 1e-5)


def test_CX_Rz_CY():
    gates = ['H', 'H', 'H', 'CX', 'Rz', 'CY']
    qubits = [[0], [1], [2], [0, 1], [1], [1, 2]]
    param_inds = [[], [], [], None, [0], []]

    param_to_st = qujax.get_params_to_statetensor_func(gates, qubits, param_inds)
    st = param_to_st(jnp.array(0.1))

    true_sv = jnp.array([0.34920055 - 0.05530793j, 0.34920055 - 0.05530793j,
                         0.05530793 - 0.34920055j, -0.05530793 + 0.34920055j,
                         0.34920055 - 0.05530793j, 0.34920055 - 0.05530793j,
                         0.05530793 - 0.34920055j, -0.05530793 + 0.34920055j], dtype='complex64')

    assert st.size == true_sv.size
    assert jnp.all(jnp.abs(st.flatten() - true_sv) < 1e-5)


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



from jax import numpy as jnp, jit, grad, random

import qujax


def test_bitstring_expectation():
    n_qubits = 4

    gates = ['H'] * n_qubits \
            + ['Ry'] * n_qubits + ['Rz'] * n_qubits \
            + ['CX'] * (n_qubits - 1) \
            + ['Ry'] * n_qubits + ['Rz'] * n_qubits
    qubits = [[i] for i in range(n_qubits)] * 3 \
             + [[i, i + 1] for i in range(n_qubits - 1)] \
             + [[i] for i in range(n_qubits)] * 2
    param_inds = [[]] * n_qubits \
                 + [[i] for i in range(n_qubits * 2)] \
                 + [[]] * (n_qubits - 1) \
                 + [[i] for i in range(n_qubits * 2, n_qubits * 4)]

    param_to_st = qujax.get_params_to_statetensor_func(gates, qubits, param_inds)

    n_params = n_qubits * 4
    params = random.uniform(random.PRNGKey(0), shape=(n_params,))

    costs = random.normal(random.PRNGKey(1), shape=(2 ** n_qubits,))

    def st_to_expectation(statetensor):
        probs = jnp.square(jnp.abs(statetensor.flatten()))
        return jnp.sum(costs * probs)

    param_to_expectation = lambda p: st_to_expectation(param_to_st(p))

    expectation = param_to_expectation(params)
    expectation_jit = jit(param_to_expectation)(params)

    assert expectation.shape == ()
    assert expectation.dtype == 'float32'
    assert jnp.abs(-0.97042876 - expectation) < 1e-5
    assert jnp.abs(-0.97042876 - expectation_jit) < 1e-5

    expectation_grad = grad(param_to_expectation)(params)
    expectation_grad_jit = jit(grad(param_to_expectation))(params)

    true_expectation_grad = jnp.array([5.1673526e-01, 1.2618620e+00, 5.1392573e-01,
                                       1.5056899e+00, 4.3226164e-02, 3.4227133e-02,
                                       8.1762001e-02, 7.7345759e-01, 5.1567715e-01,
                                       -3.1131029e-01, -1.7132770e-01, -6.6244489e-01,
                                       9.3626760e-08, -4.6813380e-08, -2.3406690e-08,
                                       -9.3626760e-08])

    assert expectation_grad.shape == (n_params,)
    assert expectation_grad.dtype == 'float32'
    assert jnp.all(jnp.abs(expectation_grad - true_expectation_grad) < 1e-5)
    assert jnp.all(jnp.abs(expectation_grad_jit - true_expectation_grad) < 1e-5)


def test_ZZ_X():
    n_qubits = 5

    gate_str_seq_seq = [['Z', 'Z']] * (n_qubits - 1) + [['X']] * n_qubits
    coefs = random.normal(random.PRNGKey(0), shape=(len(gate_str_seq_seq),))

    qubit_inds_seq = [[i, i + 1] for i in range(n_qubits - 1)] + [[i] for i in range(n_qubits)]
    st_to_exp = qujax.get_statetensor_to_expectation_func(gate_str_seq_seq,
                                                          qubit_inds_seq,
                                                          coefs)

    state = random.uniform(random.PRNGKey(0), shape=(2 ** n_qubits,)) * 2
    state /= jnp.linalg.norm(state)
    st_in = state.reshape((2,) * n_qubits)

    jax_exp = st_to_exp(st_in)
    jax_exp_jit = jit(st_to_exp)(st_in)

    assert jnp.abs(-0.23738188 - jax_exp) < 1e-5
    assert jnp.abs(-0.23738188 - jax_exp_jit) < 1e-5

    st_to_samp_exp = qujax.get_statetensor_to_sampled_expectation_func(gate_str_seq_seq,
                                                                       qubit_inds_seq,
                                                                       coefs)
    jax_samp_exp = st_to_samp_exp(st_in, random.PRNGKey(1), 1000)
    jax_samp_exp_jit = jit(st_to_samp_exp, static_argnums=2)(st_in, random.PRNGKey(2), 1000)
    assert jnp.abs(-0.23738188 - jax_samp_exp) < 1e-1
    assert jnp.abs(-0.23738188 - jax_samp_exp_jit) < 1e-1


def test_sampling():
    target_pmf = jnp.array([0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0])
    target_pmf /= target_pmf.sum()

    target_st = jnp.sqrt(target_pmf).reshape((2,) * int(jnp.log2(target_pmf.size)))

    n_samps = 7

    sample_ints = qujax.sample_integers(random.PRNGKey(0), target_st, n_samps)
    assert sample_ints.shape == (n_samps,)
    assert all(target_pmf[sample_ints] > 0)

    sample_bitstrings = qujax.sample_bitstrings(random.PRNGKey(0), target_st, n_samps)
    assert sample_bitstrings.shape == (n_samps, int(jnp.log2(target_pmf.size)))
    assert all(qujax.bitstrings_to_integers(sample_bitstrings) == sample_ints)

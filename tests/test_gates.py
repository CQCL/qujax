from qujax import gates, check_unitary


def test_gates():
    for g_str, g in gates.__dict__.items():
        if g_str[0] != '_' and g_str != 'jnp':
            check_unitary(g_str)
            check_unitary(g)

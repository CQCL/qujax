from qujax import check_unitary, gates


def test_gates():
    for g_str, g in gates.__dict__.items():
        # Exclude elements in jax.gates namespace which are not gates
        if g_str[0] != "_" and g_str not in ("jax", "jnp"):
            check_unitary(g_str)
            check_unitary(g)

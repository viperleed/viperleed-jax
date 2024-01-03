from jax import config
config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax

_REDUCED_GAUNT_COEFFICIENTS = jnp.load("gaunt_coefficients.npy",
                                       allow_pickle=False)


def fetch_stored(l1, l2, l3, m1, m2, m3):
    selection_rule_m = m1 + m2 + m3 == 0
    return jax.lax.cond(
        selection_rule_m,
        (lambda l1, l2, l3, m1, m2: 
         _REDUCED_GAUNT_COEFFICIENTS[l1, l2, l3, m1, m2]),
        lambda l1, l2, l3, m1, m2: 0.0,
        l1, l2, l3, m1, m2,
    )

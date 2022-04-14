import numpy as np
import jax.numpy as jnp
import jax
from jax import jit
from jax import vmap
from jax.experimental import stax


def HFDS(L, N, H):
    N_up = int(N/2.)


    init_b , eval_b = stax.serial(stax.Dense(L), stax.Relu,
                                  stax.Dense(int(L)), stax.Relu,
                                  stax.Dense(N+H), stax.Relu,
                                  stax.Dense(N+H), stax.Tanh)

    def init_fun(rng, input_shape):
        output_shape = input_shape[:-1] + (1,) #(batch, 1)
        init_parameters_det = 2*(np.random.rand(L, N + H)-0.5)
        init_params_bs = []
        for i in range(H):
            _ ,init_parameters_b = init_b(rng, input_shape)
            init_params_bs.append(init_parameters_b)
        return output_shape, [init_parameters_det, init_params_bs]

    @jit
    def occupation_constructor(inputs):
        """
        constructs helper matrix (N X L). For each row i we put a 1 in the column j, where j is the
        mode occupied by particle i. Then the wave function is det(helper * V).

        Function contains two loops. This is the slow part in Jax
        """
        helper = np.zeros( inputs.shape[:-1] + (N + 1, L))
        inputs = (inputs/2. + N_up/2.).astype(int)
        helper = vmap_helper_constructor(helper, inputs)

        return helper

    @jit
    def helper_constructor(helper, inputs):
        for i in range(L):  # loop through modes
            particle_index = jnp.array(inputs[i] + int(i // (L/2.)) * N_up * jnp.array(inputs[i]/(inputs[i]-0.00001), int), int)
            helper = jax.ops.index_add(helper, jax.ops.index[ particle_index, i], 1)
        return helper[ 1:, :]

    vmap_helper_constructor = jit(vmap(helper_constructor, in_axes = (0,0)))

    @jit
    def apply_fun(params, inputs, **kwargs):
        orbitals = params[0]
        b_params = params[1]
        helper = occupation_constructor(inputs)
        occupations = 2*(jnp.sum(helper, axis = -2)-0.5)
        matrix = jnp.dot(helper, orbitals)
        for i in range(H):
            last_row = jnp.expand_dims(eval_b(b_params[i], occupations), axis = -2)
            matrix = jnp.concatenate((matrix, last_row), axis = -2)
        (sgn,logdet) = jnp.linalg.slogdet(matrix)
        return jnp.log(sgn + 0j) + logdet

    return init_fun, apply_fun


#site particle number n_i
def number(local_dim):
    n = np.eye(local_dim)
    n[0,0] = 0
    return n

#density-density operator
def double_occupancy(local_dim):
    nn = np.eye((local_dim)**2)
    for i in range(local_dim):
        nn[i,i] = 0
    for i in range(local_dim-1):
        nn[(i+1)*(local_dim), (i+1)*(local_dim)] = 0

    return nn

#Transition operator c^\dag_i + c_j + C.C.
def transition(local_dim):
    T = np.zeros(((local_dim) ** 2, (local_dim) ** 2))

    for i in range(local_dim-1):
        T[(i + 1) * (local_dim), (i + 1)] = -1
    T = T + np.transpose(T)
    return T
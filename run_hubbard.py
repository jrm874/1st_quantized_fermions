import numpy as np
import argparse
import time as t
import sys
#sys.path.append('/mnt/home/jrmoreno/netket_davelopment/netket-master')
from jax.experimental import stax

import netket as nk
from netket.hilbert.hubbard import Hubbard
from wavefunction import determinant, slater_RBM, single_generalized_slave,\
    single_generalized_slave_interpretable_03, single_generalized_slave_interpretable_04, CoshLayer, SumLayerInt,\
    generalized_slave

parser = argparse.ArgumentParser()
#side length
parser.add_argument('side_length', default = 4)
#number of fermions
parser.add_argument('num_fermions', default = 8)
#Coupling constant
parser.add_argument('coupling_constant', default = 1.)
#number of iterations
parser.add_argument('epoch', default = 1000)
print (nk.__file__)

args = parser.parse_args()

L = int(args.side_length)
N_up = int(float(args.num_fermions)/2.)
N_down = int(float(args.num_fermions)/2.)
U = float(args.coupling_constant)
epoch_num = int(args.epoch)
N_hidden = 4

spin = N_up/2. #spin corresponding to each clock. Clock dimension: d = 2 * spin + 1 = N + 1 => spin = N/2
total_sz = 2 * (N_up/2. * (N_up - (L*L-1))) #total spin constraints particle number. only used in spin hilbert to provide a correct state


#define 2 disconnected 2D Torus
edges = []
#up lattice
for i in range(L * L):
    edges.append([i, L * int((i)/L) + (i + 1) % L])
for i in range(L * L):
    edges.append([i, (i+L) % (L*L)])

#down lattice
for i in range(L * L):
    edges.append([i + L**2, L * int((i)/L) + (i + 1) % L + L**2])
for i in range(L * L):
    edges.append([i + L**2, (i+L) % (L*L) + L**2])


g = nk.graph.Graph(edges = edges)

#define hilbert space with L clocks
hi = Hubbard(s = spin, graph=g , total_sz = total_sz)
_test = hi.random_state()
#_inputs = _test/2. + N_up/2. # just to viasulaize netket labling of states



#define the wave-function ansatz

#model = slater_RBM(2 * L**2, N_up + N_down, alpha = 2)
#model = determinant(2 * L**2, N_up + N_down)
#model = single_generalized_slave(2 * L**2, N_up + N_down)
#model =  single_generalized_slave_interpretable_04( 2 * L**2, N_up + N_down, d = 1)
model = generalized_slave(2 * L**2, N_up + N_down, N_hidden)
ma = nk.machine.Jax(hi, model, dtype = float) #Do not use complex type! Slogdet not implemented with complex numbers. @pfau's task
ma.init_random_parameters(seed=1232)


#_psi = ma.log_val(np.array([_test, _test]))
#print (_psi)

#-----------save and load text---------
#ma.save('_test.wf')
#_test = np.load('_test.wf', allow_pickle=False)
#params = ma.parameters
#_unflattened_test = ma.numpy_unflatten(_test, params)

sa = nk.sampler.MetropolisExchange(machine = ma, graph = g, d_max = 1, n_chains = 4)
#-----------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------OPERATORS----------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------

#site particle number n_i
n = np.eye(N_up+1)
n[0,0] = 0

#density-density operator n_i n_j
nn = np.eye((N_up+1)**2)
for i in range(N_up+1):
    nn[i,i] = 0
for i in range(N_up):
    nn[(i+1)*(N_up+1), (i+1)*(N_up+1)] = 0

#Transition operator c^\dag_i + c_j + C.C.
T = np.zeros(((N_up + 1) ** 2, (N_up + 1) ** 2))

for i in range(N_up):
    T[(i + 1) * (N_up + 1), (i + 1)] = -1
T = T + np.transpose(T)

#-----------------------------------------------------------------------------------------------------------------------
#----------------------------------------------ENERGY OPTIMIZATION------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------
#interaction edges
interaction_edges = []
for i in range(L * L):
    interaction_edges.append([i, i + L*L])
#Hamiltonian definition
operators = []
sites = []
#hoppings
for i in range(len(edges)):
    operators.append((T).tolist())
    sites.append(edges[i])
#interactions
for i in range(len(interaction_edges)):
    operators.append((U * nn).tolist())
    sites.append(interaction_edges[i])

op = nk.operator.LocalOperator(hi, operators=operators, acting_on=sites)


#Define the optimizer
optim = nk.optimizer.Sgd(ma, learning_rate=0.01)
sr = nk.optimizer.SR(ma, diag_shift=0.01)

# Create the Variational Monte Carlo instance to learn the ground state
vmc = nk.Vmc(hamiltonian=op, sampler=sa, optimizer=optim, n_samples=1000, sr=sr)

#run the optimization
vmc.run(out='Energy_L-' + str(L) + '_N-' + str(N_up + N_down) +'_U-' + str(U), n_iter=epoch_num)


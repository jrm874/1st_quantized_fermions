import numpy as np

PATH_TO_REPO = ''
import sys
sys.path.append(PATH_TO_REPO + '/netket-master')

import netket as nk
from netket.hilbert.hubbard import Hubbard
from wavefunction import *


L = 4
N_up = 4
N_down = 4
U = 10.
epoch_num = 100
N_hidden = 4

spin = N_up/2. #spin dimension
total_sz = 2 * (N_up/2. * (N_up - (L*L-1))) #total spin constraints particle number


#define lattice
edges = []
for i in range(L * L):
    edges.append([i, L * int((i)/L) + (i + 1) % L])
for i in range(L * L):
    edges.append([i, (i+L) % (L*L)])
for i in range(L * L):
    edges.append([i + L**2, L * int((i)/L) + (i + 1) % L + L**2])
for i in range(L * L):
    edges.append([i + L**2, (i+L) % (L*L) + L**2])


g = nk.graph.Graph(edges = edges)

#define hilbert space
hi = Hubbard(s = spin, graph=g , total_sz = total_sz)


#define the wave-function ansatz
model = HFDS(2 * L**2, N_up + N_down, N_hidden)
ma = nk.machine.Jax(hi, model, dtype = float)
ma.init_random_parameters(seed=1232)


sa = nk.sampler.MetropolisExchange(machine = ma, graph = g, d_max = 1, n_chains = 4)

n = number(N_up+1)
nn = double_occupancy(N_up+1)
T = transition(N_up+1)

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
vmc = nk.Vmc(hamiltonian=op, sampler=sa, optimizer=optim, n_samples=100, sr=sr)

print ('Compiling may take a few minutes')
#run the optimization
vmc.run(out='Energy_L-' + str(L) + '_N-' + str(N_up + N_down) +'_U-' + str(U), n_iter=epoch_num)


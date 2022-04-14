# __NetKet for 1st quantized fermions__


Requirements:
- **1.** Python >= 3.6.
- **2.** numpy >=1.16.
- **3.** scipy>=1.5.2.
- **4.** tqdm>=4.42.1
- **5.** numba>=0.49.0
- **6.** networkx>=2.4
- **7.** jax

The script run_hubbard.py is an example of the optimization of the
hidden fermion determinant state where the hidden sub-matrix is
parametrized by multilayer perceptrons. The physical system is a 4X4
square Hubbard model at quarter occupation with onsite repulsion U = 10.
To run just use:

```python3 run_hubbard.py```

The script wavefunction.py contains the JAX implementation of the wave
function, as well as the definition of the operators that enter the
Hamiltonian.

The VMC sampler and optimizer use the netket backend provided in this
repository.

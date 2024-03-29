# Generating Function for Projected Entangled-Pair States
### A tensor network library for two-dimensional excitation

\
peps-excitation utilizes the ground state ansatz and calculates the excitation ansatz combining the usage of generating function in two dimensions [PRX Quantum 5, 010335 (2024)](https://journals.aps.org/prxquantum/abstract/10.1103/PRXQuantum.5.010335). 

#### Requirement
- PyTorch 1.+ (see https://pytorch.org/)
- GPU is recommended to run the code

**More details can be found in the paper.**
* * *
<br>

#### Examples
**Ex. 1)** Using the one-site (C4v) symmetric iPEPS with bond dimension D=2 for generating the excited tensor of transverse-field Ising model with h=3.5.

```
python excitation_ising.py --GLOBALARGS_dtype complex128 --bond_dim 2 --chi 8 --size 11 --seed 123 --hx 3.5 --instate ex-hx35D2chi8c4v_state.json
```
Input state can be generated by [peps-torch](https://github.com/jurajHasik/tn-torch_dev). The overall size of central cluster is (2 * size + 2) * (2 * size + 2). The above comment creates a Norm and a Hami .txt files. Then run the excitation_ising_input.py file with the following comment:
```
python excitation_ising_input.py --GLOBALARGS_dtype complex128 --bond_dim 2 --chi 8 --size 11 --seed 123 --hx 3.5 --instate ex-hx35D2chi8c4v_state.json
```
And then the excited energies can be calculated.

**Ex. 2)** Heisenberg model

```
python excitation_hei.py --GLOBALARGS_dtype complex128 --bond_dim 2 --chi 8 --size 11 --seed 123 --j2 0. --instate ex-j20D2chi8c4v_state.json
```
After obtaining the Norm and Hami .txt files, run excitation_hei_input.py file with the following comment:
```
python excitation_hei_input.py --GLOBALARGS_dtype complex128 --bond_dim 2 --chi 8 --size 11 --seed 123 --j2 0. --instate ex-j20D2chi8c4v_state.json
```

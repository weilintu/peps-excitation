# peps_excitation
### A tensor network library for two-dimensional excitation
by Wei-Lin Tu

\
peps-excitation performs optimization of infinite Projected entangled-pair states (iPEPS) 
by direct energy minimization. The energy is evaluated using environments obtained 
by the corner-transfer matrix (CTM) algorithm. Afterwards, the gradients are computed by reverse-mode 
automatic differentiation (AD).

[peps-torch.readthedocs.io](https://peps-torch.readthedocs.io)

--size: the size of cluster of upper/lower (left/right) sides. The overall size is (2 * size + 2) * (2 * size + 2)

Ground state ansatz can be generated from peps-torch (https://github.com/jurajHasik/peps-torch).

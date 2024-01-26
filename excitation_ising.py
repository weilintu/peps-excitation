#python excitation_ising.py --GLOBALARGS_dtype complex128 --bond_dim 2 --chi 8 --size 11 --seed 123 --hx 3.5 --instate ex-hx35D2chi8c4v_state.json
import context
import time
import torch
import argparse
import config as cfg
from ipeps.ipeps import *
from ctm.generic.env import *
from ctm.generic.rdm import *
from ctm.generic import ctmrg_ex
from ctm.generic.ctm_projectors import *
from Norm import *
from Hami import *
from models import ising
from groups.pg import *
import groups.su2 as su2
from tn_interface import contract, einsum
from tn_interface import conj
from tn_interface import contiguous, view, permute
import numpy as np
import scipy as scipy
import scipy.linalg as linalg
import scipy.sparse.linalg as spr_linalg
import scipy.io
import unittest
import logging
log = logging.getLogger(__name__)

tStart = time.time()

# parse command line args and build necessary configuration objects
parser= cfg.get_args_parser()
# additional model-dependent arguments
parser.add_argument("--hx", type=float, default=0., help="transverse field")
parser.add_argument("--q", type=float, default=0., help="next nearest-neighbour coupling")
parser.add_argument("--size", type=int, default=10, help="effective size")
args, unknown_args = parser.parse_known_args()

cfg.configure(args)
torch.set_num_threads(args.omp_cores)
torch.manual_seed(args.seed)

# assign the model
model = ising.ISING(hx=args.hx, q=args.q)
state = read_ipeps(args.instate, vertexToSite=None)

# generate the tensor with c4v symmetry
def symmetrize(state):
    A= state.site((0,0))
    A_symm= make_c4v_symm_A1(A)
    symm_state= IPEPS({(0,0): A_symm}, vertexToSite=state.vertexToSite)
    return symm_state
state= symmetrize(state)
sitesDL=dict()
for coord,A in state.sites.items():
    dimsA = A.size()
    a = contiguous(einsum('mefgh,mabcd->eafbgchd',A,conj(A)))
    a = view(a, (dimsA[1]**2,dimsA[2]**2, dimsA[3]**2, dimsA[4]**2))
    sitesDL[coord]=a
stateDL = IPEPS(sitesDL,state.vertexToSite)

# define the function for CTMRG convergence
def ctmrg_conv_energy(state2, env, history, ctm_args=cfg.ctm_args):
    if not history:
        history=[]

    if (len(history)>0):
        old = history[:4*env.chi]
    new = []
    u,s,v = torch.svd(env.C[((0,0),(-1,-1))])
    for i in range(env.chi):
        new.append(s[i].item())
    u,s,v = torch.svd(env.C[((0,0),(1,-1))])
    for i in range(env.chi):
        new.append(s[i].item())
    u,s,v = torch.svd(env.C[((0,0),(1,-1))])
    for i in range(env.chi):
        new.append(s[i].item())
    u,s,v = torch.svd(env.C[((0,0),(1,1))])
    for i in range(env.chi):
        new.append(s[i].item())

    diff = 0.
    if (len(history)>0):
        for i in range(4*env.chi):
            history[i] = new[i]
            if (abs(old[i]-new[i])>diff):
                diff = abs(old[i]-new[i])
    else:
        for i in range(4*env.chi):
            history.append(new[i])
    history.append(diff)

    if (len(history[4*env.chi:]) > 1 and diff < ctm_args.ctm_conv_tol)\
        or len(history[4*env.chi:]) >= ctm_args.ctm_max_iter:
        log.info({"history_length": len(history[4*env.chi:]), "history": history[4*env.chi:]})
        #print (len(history[4*env.chi:]))
        return True, history
    return False, history

# calculate the ground state energy
env = ENV(args.chi, state)
init_env(state, env)

env, P, Pt, *ctm_log = ctmrg_ex.run(state, env, conv_check=ctmrg_conv_energy)
print ("E_per_site=", model.energy_1x1(state, env).item())

################Suzuki-Trotter gate for Hamiltonian################
torch.pi = torch.tensor(np.pi, dtype=torch.complex128)
# assign the wave number
kx_int = 0
ky_int = 0
kx = kx_int*torch.pi/(2*args.size+2)
ky = ky_int*torch.pi/(2*args.size+2)
print ("kx=", kx/torch.pi*(2*args.size+2))
print ("ky=", ky/torch.pi*(2*args.size+2))
SzSz = model.h2
Sx = model.h1
iden= torch.eye(2,dtype=cfg.global_args.torch_dtype,device=cfg.global_args.device).contiguous()
H_temp = -SzSz - args.hx * (torch.einsum('ij,kl->ikjl',Sx,iden)+torch.einsum('ij,kl->ikjl',iden,Sx))/4
lam = torch.tensor(0.0,dtype=cfg.global_args.torch_dtype,device=cfg.global_args.device).requires_grad_(True)
lamb = torch.tensor(1.0,dtype=cfg.global_args.torch_dtype,device=cfg.global_args.device)
iden2 = torch.einsum('ij,kl->ikjl',iden,iden)
H = iden2 + lam * H_temp
H2 = iden2

# calculate the energy per bond
rdm2x1= rdm2x1((0,0),state,env)
energy_per_site= torch.einsum('ijkl,ijkl',rdm2x1,H_temp)
print ("E_per_bond=", energy_per_site.item().real)

################Excited state algorithm################
bond_dim = args.bond_dim

################Effective norm################
B_grad = torch.zeros((len(state.sites), model.phys_dim, bond_dim, bond_dim, bond_dim, bond_dim),\
                     dtype=cfg.global_args.torch_dtype,device=cfg.global_args.device).requires_grad_(True)
if len(state.sites)==1:
    sitesB = {(0,0): B_grad[0]}
    stateB = IPEPS(sitesB, state.vertexToSite)

with torch.no_grad():
    P, Pt = Create_Projectors(state, stateDL, env, args)
C_up, T_up, C_left, T_left, C_down, T_down, C_right, T_right = Create_Norm_Env(state, stateDL, stateB, env, P, Pt, lamb, kx, ky, args)
Norm, Norm2 = Create_Norm(state, env, C_up, T_up, C_left, T_left, C_down, T_down, C_right, T_right, args)

if len(state.sites)==1:
    norm_factor = contract(Norm2[(0,0)],conj(state.site((0,0))),([0,1,2,3,4],[0,1,2,3,4]))
    Norm[(0,0)] = (Norm[(0,0)])/norm_factor
    Norm2[(0,0)] = (Norm2[(0,0)])/norm_factor
    print ("G(N)_dot_state=", contract(Norm[(0,0)],conj(state.site((0,0))),([0,1,2,3,4],[0,1,2,3,4])).item().real)
    print ("Norm_factor=", norm_factor.item().real)

    N_para = state.site((0,0)).size()[0]*state.site((0,0)).size()[1]*state.site((0,0)).size()[2]*state.site((0,0)).size()[3]*state.site((0,0)).size()[4]
    N_final = torch.zeros((N_para,N_para),\
                          dtype=cfg.global_args.torch_dtype,device=cfg.global_args.device)
    Norm_d = view(Norm[(0,0)],(N_para))
    dev_accu = torch.zeros((N_para), \
                           dtype=cfg.global_args.torch_dtype,device=cfg.global_args.device)
    for i in range(N_para):
        Norm_d[i].backward(torch.ones_like(Norm_d[i]), retain_graph=True)
        temp_d = B_grad.grad.view(N_para).clone()
        dev = temp_d.clone() - dev_accu
        dev_accu = temp_d.clone()
        N_final[i] = dev
    state_t = view(state.site((0,0)),(N_para))
    temp = contract(N_final,conj(state_t),([1],[0]))
    print ("<Norm>=", contract(temp,state_t,([0],[0])).item().real/(2*args.size+2)**2)
    N_final = N_final/(2*args.size+2)**2

    N_export = N_final.detach().cpu().numpy()
    N_export = N_export.reshape(N_para**2)
    str1 = " ".join(str(e) for e in N_export)
    f=open('Norm_kx'+str(kx_int)+'ky'+str(ky_int)+'.txt', 'w')
    f.write(str1)
    f.close()


tEnd = time.time()
print ("time_Norm=", tEnd - tStart)
################Effective H################
B_grad = torch.zeros((len(state.sites), model.phys_dim, bond_dim, bond_dim, bond_dim, bond_dim),\
                     dtype=cfg.global_args.torch_dtype,device=cfg.global_args.device).requires_grad_(True)
if len(state.sites)==1:
    sitesB = {(0,0): B_grad[0]}
    stateB = IPEPS(sitesB, state.vertexToSite)

C_up, T_up, C_left, T_left, C_down, T_down, C_right, T_right = Create_Hami_Env(state, stateDL, stateB, env, P, Pt, lamb, H, H2, kx, ky, args)
Hami, Hami2 = Create_Hami(state, env, C_up, T_up, C_left, T_left, C_down, T_down, C_right, T_right, H, H2, args)


if len(state.sites)==1:
    Hami[(0,0)] = Hami[(0,0)]/norm_factor
    Hami2[(0,0)] = Hami2[(0,0)]/norm_factor
    print ("G(H)_dot_state=", contract(Hami[(0,0)],conj(state.site((0,0))),([0,1,2,3,4],[0,1,2,3,4])).item().real)

    N_para = state.site((0,0)).size()[0]*state.site((0,0)).size()[1]*state.site((0,0)).size()[2]*state.site((0,0)).size()[3]*state.site((0,0)).size()[4]
    H_final = torch.zeros((N_para,N_para),\
                          dtype=cfg.global_args.torch_dtype,device=cfg.global_args.device)
    Hami_d = view(Hami[(0,0)],(N_para))
    dev_accu = torch.zeros((N_para), \
                           dtype=cfg.global_args.torch_dtype,device=cfg.global_args.device)
    for i in range(N_para):
        g = torch.autograd.grad(Hami_d[i].real,lam,create_graph=True)
        g[0].real.backward(retain_graph=True)
        temp_d = B_grad.grad.view(N_para).clone()
        dev = temp_d.clone() - dev_accu
        dev_accu = temp_d.clone()
        H_final[i] = dev
    state_t = view(state.site((0,0)),(N_para))
    temp = contract(H_final,conj(state_t),([1],[0]))
    print ("<Hami>=", 2*contract(temp,state_t,([0],[0])).item().real/(2*args.size+2)**3/(2*args.size+1))
    H_final = 2*H_final/(2*args.size+2)**3/(2*args.size+1)

    H_export = H_final.detach().cpu().numpy()
    H_export = H_export.reshape(N_para**2)
    str1 = " ".join(str(e) for e in H_export)
    f=open('Hami_kx'+str(kx_int)+'ky'+str(ky_int)+'.txt', 'w')
    f.write(str1)
    f.close()


tEnd = time.time()
print ("time_Hami=", tEnd - tStart)

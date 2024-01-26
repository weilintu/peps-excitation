#python excitation_hei_input.py --GLOBALARGS_dtype complex128 --bond_dim 2 --chi 8 --size 11 --seed 123 --j2 0. --instate ex-j20D2chi8c4v_state.json
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
from models import j1j2
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
parser.add_argument("--j1", type=float, default=1., help="nearest-neighbour coupling")
parser.add_argument("--j2", type=float, default=0., help="next nearest-neighbour coupling")
parser.add_argument("--size", type=int, default=10, help="effective size")
args, unknown_args = parser.parse_known_args()

cfg.configure(args)
torch.set_num_threads(args.omp_cores)
torch.manual_seed(args.seed)

# assign the model
model = j1j2.J1J2(j1=args.j1, j2=args.j2)
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
print ("E_per_site=", model.energy_2x2_1site_BP(state, env).item())

################Suzuki-Trotter gate for Hamiltonian################
torch.pi = torch.tensor(np.pi, dtype=torch.complex128)
kx_int = 0
ky_int = 0
kx = kx_int*torch.pi/(2*args.size+2)
ky = ky_int*torch.pi/(2*args.size+2)
print ("kx=", kx/torch.pi*(2*args.size+2))
print ("ky=", ky/torch.pi*(2*args.size+2))
SS = model.SS_rot
iden= torch.eye(2,dtype=cfg.global_args.torch_dtype,device=cfg.global_args.device).contiguous()
H_temp = args.j1 * SS
lam = torch.tensor(0.0,dtype=cfg.global_args.torch_dtype,device=cfg.global_args.device).requires_grad_(True)
lamb = torch.tensor(1.0,dtype=cfg.global_args.torch_dtype,device=cfg.global_args.device)
iden2 = torch.einsum('ij,kl->ikjl',iden,iden)
H = iden2 + lam * H_temp
H2 = iden2

# calculate the energy per bond
rdm2x1= rdm2x1((0,0),state,env)
energy_per_site= torch.einsum('ijkl,ijkl',rdm2x1,H_temp)
print ("E_per_bond=", 2*energy_per_site.item().real)

################Excited state################
bond_dim = args.bond_dim

################Effective norm################
B_grad = torch.zeros((len(state.sites), model.phys_dim, bond_dim, bond_dim, bond_dim, bond_dim),\
                     dtype=cfg.global_args.torch_dtype,device=cfg.global_args.device).requires_grad_(True)
if len(state.sites)==1:
    sitesB = {(0,0): B_grad[0]}
    stateB = IPEPS(sitesB, state.vertexToSite)
with torch.no_grad():
    P, Pt = Create_Projectors(state, stateDL, env, args)

eig_size = 1
if len(state.sites)==1:
    N_para = state.site((0,0)).size()[0]*state.site((0,0)).size()[1]*state.site((0,0)).size()[2]*state.site((0,0)).size()[3]*state.site((0,0)).size()[4]

    N_final = np.zeros((N_para,N_para), dtype=np.complex128)
    
    filename1 = 'Norm_kx'+str(kx_int)+'ky'+str(ky_int)+'.txt'
    with open(filename1, 'r') as file_to_read2:
        lines2 = file_to_read2.readline()
        q_tmp = [complex(k) for k in lines2.replace('i', 'j').split()]
        for i in range(N_para):
            for j in range(N_para):
                N_final[i][j]=q_tmp[i*N_para+j]

    N_final = N_final+conj(N_final.transpose(1,0))
    N_cmn = N_final.copy()


tEnd = time.time()
print ("time_Norm=", tEnd - tStart)
################Effective H################
if len(state.sites)==1:
    N_para = state.site((0,0)).size()[0]*state.site((0,0)).size()[1]*state.site((0,0)).size()[2]*state.site((0,0)).size()[3]*state.site((0,0)).size()[4]

    H_final = np.zeros((N_para,N_para), dtype=np.complex128)

    filename1 = 'Hami_kx'+str(kx_int)+'ky'+str(ky_int)+'.txt'
    with open(filename1, 'r') as file_to_read2:
        lines2 = file_to_read2.readline()
        q_tmp = [complex(k) for k in lines2.replace('i', 'j').split()]
        for i in range(N_para):
            for j in range(N_para):
                H_final[i][j]=q_tmp[i*N_para+j]

    H_final = H_final+conj(H_final.transpose(1,0))
    H_cmn = H_final.copy()


tEnd = time.time()
print ("time_Hami=", tEnd - tStart)
################Eigenvalue problem################
e, v = np.linalg.eig(N_cmn)
idx = np.argsort(-e.real)   
e = e[idx]
v = v[:,idx]

# adjust the size of reduced Hilbert space
eig_size = 6
vt = np.zeros((N_para,eig_size), dtype=np.complex128)
for i in range(eig_size):
    vt[:,i] = v[:,i]

N_cmn2 = np.matmul(N_cmn, vt)
N_cmn2 = np.matmul(np.transpose(np.conj(vt)), N_cmn2)
H_cmn2 = np.matmul(H_cmn, vt)
H_cmn2 = np.matmul(np.transpose(np.conj(vt)), H_cmn2)

N_cmn_inv = linalg.pinvh(N_cmn2)#, cond=0.000001, rcond=0.000001)
ef, vf = np.linalg.eig(np.matmul(N_cmn_inv,H_cmn2))
idx = np.argsort(ef)   
ef = ef[idx]
vf = vf[:,idx]
    

if kx == ky == 0:
    print ("E_lowest_ex=",(2*args.size+2)*(2*args.size+1)*(ef[1]-2*energy_per_site.item().real))
else:
    print ("E_lowest_ex=",(2*args.size+2)*(2*args.size+1)*(ef[0]-2*energy_per_site.item().real))


tEnd = time.time()
print ("time_ener=", tEnd - tStart)

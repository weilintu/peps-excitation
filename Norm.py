import context
import time
import torch
import argparse
import config as cfg
from ctm.generic.ctm_projectors import *
from ipeps.ipeps import *
from tn_interface import contract, einsum
from tn_interface import conj
from tn_interface import contiguous, view, permute

def Create_Projectors(state, stateDL, env, args):
    if cfg.ctm_args.projector_method=='4X4':
        ctm_get_projectors=ctm_get_projectors_4x4
    elif cfg.ctm_args.projector_method=='4X2':
        ctm_get_projectors=ctm_get_projectors_4x2
    else:
        raise ValueError("Invalid Projector method: "+str(cfg.ctm_args.projector_method))
    P = dict()
    Pt = dict()

    ctm_env_ex2 = env.clone()
    for i in range(args.size+1):
        for coord in stateDL.sites.keys():
            P[(i,coord,(-1,0))], Pt[(i,coord,(-1,0))] = ctm_get_projectors((-1,0), coord, stateDL, ctm_env_ex2, cfg.ctm_args, cfg.global_args)
            P[(i,coord,(1,0))], Pt[(i,coord,(1,0))] = ctm_get_projectors((1,0), coord, stateDL, ctm_env_ex2, cfg.ctm_args, cfg.global_args)
        for coord in stateDL.sites.keys():
            for direction in cfg.ctm_args.ctm_move_sequence:
                if direction==(-1,0):
                    ################edge tensor################
                    vec_coord = (-args.size+i,0)
                    new_coord = state.vertexToSite((coord[0]+vec_coord[0], coord[1]+vec_coord[1]))
                    vec = (0,-1)
                    
                    coord_shift_up= stateDL.vertexToSite((new_coord[0]+vec[0], new_coord[1]+vec[1]))
                    coord_shift_down= stateDL.vertexToSite((new_coord[0]-vec[0], new_coord[1]-vec[1]))
                    P2 = view(P[(i,new_coord,direction)], (env.chi,stateDL.site(coord_shift_down).size()[0],env.chi))
                    Pt2 = view(Pt[(i,new_coord,direction)], (env.chi,stateDL.site(new_coord).size()[2],env.chi))
                    P1 = view(P[(i,coord_shift_up,direction)], (env.chi,stateDL.site(new_coord).size()[0],env.chi))
                    Pt1 = view(Pt[(i,coord_shift_up,direction)], (env.chi,stateDL.site(coord_shift_up).size()[2],env.chi))

                    nT = contract(P1, ctm_env_ex2.T[(new_coord,direction)],([0],[0]))
                    dimsA = state.site(new_coord).size()
                    Aket = state.site(new_coord)
                    DL = torch.reshape(einsum('mefgh,mabcd->eafbgchd',Aket,conj(state.site(new_coord))),\
                              (dimsA[1]**2, dimsA[2]**2, dimsA[3]**2, dimsA[4]**2))
                    nT = contract(nT, DL,([0,3],[0,1]))
                    nT = contract(nT, Pt2,([1,2],[0,1]))
                    tempT2 = contiguous(permute(nT, (0,2,1)))
                    tempT2 = tempT2/tempT2.abs().max()
                    
                    vec_coord = (vec_coord[0]+1,vec_coord[1])
                    new_coord = state.vertexToSite((coord[0]+vec_coord[0], coord[1]+vec_coord[1]))
                    
                    ctm_env_ex2.T[(new_coord,direction)] = tempT2.clone()

                    ################corner tensor################
                    vec_coord_u = (-args.size+i,-args.size)
                    new_coord_u = state.vertexToSite((coord[0]+vec_coord_u[0], coord[1]+vec_coord_u[1]))
                    coord_shift_up= stateDL.vertexToSite((new_coord_u[0]+vec[0], new_coord_u[1]+vec[1]))
                    coord_shift_down= stateDL.vertexToSite((new_coord_u[0]-vec[0], new_coord_u[1]-vec[1]))
                    P2 = view(P[(i,new_coord_u,direction)], (env.chi,stateDL.site(coord_shift_down).size()[0],env.chi))
                    Pt2 = view(Pt[(i,new_coord_u,direction)], (env.chi,stateDL.site(new_coord_u).size()[2],env.chi))
                    P1 = view(P[(i,coord_shift_up,direction)], (env.chi,stateDL.site(new_coord_u).size()[0],env.chi))
                    Pt1 = view(Pt[(i,coord_shift_up,direction)], (env.chi,stateDL.site(coord_shift_up).size()[2],env.chi))
                    nC1 = contract(ctm_env_ex2.C[(new_coord_u,(-1,-1))], ctm_env_ex2.T[(new_coord_u,(0,-1))],([1],[0]))
                    nC1 = contract(Pt1, nC1,([0,1],[0,1]))
                    tempT2 = nC1/nC1.abs().max()

                    vec_coord = (vec_coord_u[0]+1,vec_coord_u[1])
                    new_coord = state.vertexToSite((coord[0]+vec_coord[0], coord[1]+vec_coord[1]))
                    
                    ctm_env_ex2.C[(new_coord,(-1,-1))] = tempT2.clone()

                    vec_coord_d = (-args.size+i,args.size+1)
                    new_coord_d = state.vertexToSite((coord[0]+vec_coord_d[0], coord[1]+vec_coord_d[1]))
                    coord_shift_up= stateDL.vertexToSite((new_coord_d[0]+vec[0], new_coord_d[1]+vec[1]))
                    coord_shift_down= stateDL.vertexToSite((new_coord_d[0]-vec[0], new_coord_d[1]-vec[1]))
                    P2 = view(P[(i,new_coord_d,direction)], (env.chi,stateDL.site(coord_shift_down).size()[0],env.chi))
                    Pt2 = view(Pt[(i,new_coord_d,direction)], (env.chi,stateDL.site(new_coord_d).size()[2],env.chi))
                    P1 = view(P[(i,coord_shift_up,direction)], (env.chi,stateDL.site(new_coord_d).size()[0],env.chi))
                    Pt1 = view(Pt[(i,coord_shift_up,direction)], (env.chi,stateDL.site(coord_shift_up).size()[2],env.chi))
                    nC2 = contract(ctm_env_ex2.C[(new_coord_d,(-1,1))], ctm_env_ex2.T[(new_coord_d,(0,1))],([1],[1]))
                    nC2 = contract(P2, nC2,([0,1],[0,1]))
                    tempT2 = nC2/nC2.abs().max()

                    vec_coord = (vec_coord_d[0]+1,vec_coord_d[1])
                    new_coord = state.vertexToSite((coord[0]+vec_coord[0], coord[1]+vec_coord[1]))
                    
                    ctm_env_ex2.C[(new_coord,(-1,1))] = tempT2.clone()

                elif direction==(1,0):
                    ################edge tensor################
                    vec_coord = (args.size-i,0)
                    new_coord = state.vertexToSite((coord[0]+vec_coord[0], coord[1]+vec_coord[1]))
                    vec = (0,1)
                    
                    coord_shift_down = stateDL.vertexToSite((new_coord[0]+vec[0], new_coord[1]+vec[1]))
                    coord_shift_up = stateDL.vertexToSite((new_coord[0]-vec[0], new_coord[1]-vec[1]))
                    P2 = view(P[(i,new_coord,direction)], (env.chi,stateDL.site(coord_shift_up).size()[2],env.chi))
                    Pt2 = view(Pt[(i,new_coord,direction)], (env.chi,stateDL.site(new_coord).size()[0],env.chi))
                    P1 = view(P[(i,coord_shift_down,direction)], (env.chi,stateDL.site(new_coord).size()[2],env.chi))
                    Pt1 = view(Pt[(i,coord_shift_down,direction)], (env.chi,stateDL.site(coord_shift_down).size()[0],env.chi))

                    nT = contract(Pt2, ctm_env_ex2.T[(new_coord,direction)],([0],[0]))
                    dimsA = state.site(new_coord).size()
                    Aket = state.site(new_coord)
                    DL = torch.reshape(einsum('mefgh,mabcd->eafbgchd',Aket,conj(state.site(new_coord))),\
                              (dimsA[1]**2, dimsA[2]**2, dimsA[3]**2, dimsA[4]**2))
                    nT = contract(nT, DL,([0,2],[0,3]))
                    nT = contract(nT, P1,([1,3],[0,1]))
                    tempT2 = contiguous(nT)
                    tempT2 = tempT2/tempT2.abs().max()
                    
                    vec_coord = (vec_coord[0]-1,vec_coord[1])
                    new_coord = state.vertexToSite((coord[0]+vec_coord[0], coord[1]+vec_coord[1]))
                    
                    ctm_env_ex2.T[(new_coord,direction)] = tempT2.clone()

                    ################corner tensor################
                    vec_coord_u = (args.size-i+1,-args.size)
                    new_coord_u = state.vertexToSite((coord[0]+vec_coord_u[0], coord[1]+vec_coord_u[1]))
                    coord_shift_down = stateDL.vertexToSite((new_coord_u[0]+vec[0], new_coord_u[1]+vec[1]))
                    coord_shift_up = stateDL.vertexToSite((new_coord_u[0]-vec[0], new_coord_u[1]-vec[1]))
                    P2 = view(P[(i,new_coord_u,direction)], (env.chi,stateDL.site(coord_shift_up).size()[2],env.chi))
                    Pt2 = view(Pt[(i,new_coord_u,direction)], (env.chi,stateDL.site(new_coord_u).size()[0],env.chi))
                    P1 = view(P[(i,coord_shift_down,direction)], (env.chi,stateDL.site(new_coord_u).size()[2],env.chi))
                    Pt1 = view(Pt[(i,coord_shift_down,direction)], (env.chi,stateDL.site(coord_shift_down).size()[0],env.chi))
                    nC2 = contract(ctm_env_ex2.C[(new_coord_u,(1,-1))], ctm_env_ex2.T[(new_coord_u,(0,-1))],([0],[2]))
                    nC2 = contract(nC2, P2,([0,2],[0,1]))
                    tempT2 = nC2/nC2.abs().max()

                    vec_coord = (vec_coord_u[0]-1,vec_coord_u[1])
                    new_coord = state.vertexToSite((coord[0]+vec_coord[0], coord[1]+vec_coord[1]))
                    
                    ctm_env_ex2.C[(new_coord,(1,-1))] = tempT2.clone()
                    
                    vec_coord_d = (args.size-i+1,args.size+1)
                    new_coord_d = state.vertexToSite((coord[0]+vec_coord_d[0], coord[1]+vec_coord_d[1]))
                    coord_shift_down = stateDL.vertexToSite((new_coord_d[0]+vec[0], new_coord_d[1]+vec[1]))
                    coord_shift_up = stateDL.vertexToSite((new_coord_d[0]-vec[0], new_coord_d[1]-vec[1]))
                    P2 = view(P[(i,new_coord_d,direction)], (env.chi,stateDL.site(coord_shift_up).size()[2],env.chi))
                    Pt2 = view(Pt[(i,new_coord_d,direction)], (env.chi,stateDL.site(new_coord_d).size()[0],env.chi))
                    P1 = view(P[(i,coord_shift_down,direction)], (env.chi,stateDL.site(new_coord_d).size()[2],env.chi))
                    Pt1 = view(Pt[(i,coord_shift_down,direction)], (env.chi,stateDL.site(coord_shift_down).size()[0],env.chi))
                    nC1 = contract(ctm_env_ex2.C[(new_coord_d,(1,1))], ctm_env_ex2.T[(new_coord_d,(0,1))],([1],[2]))
                    nC1 = contract(Pt1, nC1,([0,1],[0,1]))
                    tempT2 = nC1/nC1.abs().max()

                    vec_coord = (vec_coord_d[0]-1,vec_coord_d[1])
                    new_coord = state.vertexToSite((coord[0]+vec_coord[0], coord[1]+vec_coord[1]))
                    
                    ctm_env_ex2.C[(new_coord,(1,1))] = tempT2.clone()

    ctm_env_ex2 = env.clone()
    for i in range(args.size+1):
        for coord in stateDL.sites.keys():
            P[(i,coord,(0,1))], Pt[(i,coord,(0,1))] = ctm_get_projectors((0,1), coord, stateDL, ctm_env_ex2, cfg.ctm_args, cfg.global_args)
            P[(i,coord,(0,-1))], Pt[(i,coord,(0,-1))] = ctm_get_projectors((0,-1), coord, stateDL, ctm_env_ex2, cfg.ctm_args, cfg.global_args)
        for coord in stateDL.sites.keys():
            for direction in cfg.ctm_args.ctm_move_sequence:
                if direction==(0,-1):
                    ################edge tensor################
                    vec_coord = (0,-args.size+i)
                    new_coord = state.vertexToSite((coord[0]+vec_coord[0], coord[1]+vec_coord[1]))
                    vec = (1,0)
                    
                    coord_shift_left= stateDL.vertexToSite((new_coord[0]-vec[0], new_coord[1]-vec[1]))
                    coord_shift_right = stateDL.vertexToSite((new_coord[0]+vec[0], new_coord[1]+vec[1]))
                    P2 = view(P[(i,new_coord,direction)], (env.chi,stateDL.site(coord_shift_left).size()[3],env.chi))
                    Pt2 = view(Pt[(i,new_coord,direction)], (env.chi,stateDL.site(new_coord).size()[1],env.chi))
                    P1 = view(P[(i,coord_shift_right,direction)], (env.chi,stateDL.site(new_coord).size()[3],env.chi))
                    Pt1 = view(Pt[(i,coord_shift_right,direction)], (env.chi,stateDL.site(coord_shift_right).size()[1],env.chi))
                    
                    nT = contract(Pt2, ctm_env_ex2.T[(new_coord,direction)], ([0],[0]))
                    dimsA = state.site(new_coord).size()
                    Aket = state.site(new_coord)
                    DL = torch.reshape(einsum('mefgh,mabcd->eafbgchd',Aket,conj(state.site(new_coord))),\
                              (dimsA[1]**2, dimsA[2]**2, dimsA[3]**2, dimsA[4]**2))
                    nT = contract(nT, DL,([0,2],[1,0]))
                    nT = contract(nT, P1,([1,3],[0,1]))
                    tempT2 = contiguous(nT)
                    tempT2 = tempT2/tempT2.abs().max()
                    
                    vec_coord = (vec_coord[0],vec_coord[1]+1)
                    new_coord = state.vertexToSite((coord[0]+vec_coord[0], coord[1]+vec_coord[1]))
                    
                    ctm_env_ex2.T[(new_coord,direction)] = tempT2.clone()

                    ################corner tensor################
                    vec_coord_l = (-args.size,-args.size+i)
                    new_coord_l = state.vertexToSite((coord[0]+vec_coord_l[0], coord[1]+vec_coord_l[1]))
                    coord_shift_left= stateDL.vertexToSite((new_coord_l[0]-vec[0], new_coord_l[1]-vec[1]))
                    coord_shift_right = stateDL.vertexToSite((new_coord_l[0]+vec[0], new_coord_l[1]+vec[1]))
                    P2 = view(P[(i,new_coord_l,direction)], (env.chi,stateDL.site(coord_shift_left).size()[3],env.chi))
                    Pt2 = view(Pt[(i,new_coord_l,direction)], (env.chi,stateDL.site(new_coord_l).size()[1],env.chi))
                    P1 = view(P[(i,coord_shift_right,direction)], (env.chi,stateDL.site(new_coord_l).size()[3],env.chi))
                    Pt1 = view(Pt[(i,coord_shift_right,direction)], (env.chi,stateDL.site(coord_shift_right).size()[1],env.chi))
                    nC2 = contract(ctm_env_ex2.C[(new_coord_l,(-1,-1))], ctm_env_ex2.T[(new_coord_l,(-1,0))],([0],[0]))
                    nC2 = contract(nC2, P2,([0,2],[0,1]))
                    tempT2 = nC2/nC2.abs().max()

                    vec_coord = (vec_coord_l[0],vec_coord_l[1]+1)
                    new_coord = state.vertexToSite((coord[0]+vec_coord[0], coord[1]+vec_coord[1]))
                    
                    ctm_env_ex2.C[(new_coord,(-1,-1))] = tempT2.clone()
                    
                    vec_coord_r = (args.size+1,-args.size+i)
                    new_coord_r = state.vertexToSite((coord[0]+vec_coord_r[0], coord[1]+vec_coord_r[1]))
                    coord_shift_left= stateDL.vertexToSite((new_coord_r[0]-vec[0], new_coord_r[1]-vec[1]))
                    coord_shift_right = stateDL.vertexToSite((new_coord_r[0]+vec[0], new_coord_r[1]+vec[1]))
                    P2 = view(P[(i,new_coord_r,direction)], (env.chi,stateDL.site(coord_shift_left).size()[3],env.chi))
                    Pt2 = view(Pt[(i,new_coord_r,direction)], (env.chi,stateDL.site(new_coord_r).size()[1],env.chi))
                    P1 = view(P[(i,coord_shift_right,direction)], (env.chi,stateDL.site(new_coord_r).size()[3],env.chi))
                    Pt1 = view(Pt[(i,coord_shift_right,direction)], (env.chi,stateDL.site(coord_shift_right).size()[1],env.chi))
                    nC1 = contract(ctm_env_ex2.C[(new_coord_r,(1,-1))], ctm_env_ex2.T[(new_coord_r,(1,0))],([1],[0]))
                    nC1 = contract(Pt1, nC1,([0,1],[0,1]))
                    tempT2 = nC1/nC1.abs().max()

                    vec_coord = (vec_coord_r[0],vec_coord_r[1]+1)
                    new_coord = state.vertexToSite((coord[0]+vec_coord[0], coord[1]+vec_coord[1]))
                    
                    ctm_env_ex2.C[(new_coord,(1,-1))] = tempT2.clone()

                elif direction==(0,1):
                    ################edge tensor################
                    vec_coord = (0,args.size-i)
                    new_coord = state.vertexToSite((coord[0]+vec_coord[0], coord[1]+vec_coord[1]))
                    vec = (-1,0)
                    
                    coord_shift_right= stateDL.vertexToSite((new_coord[0]-vec[0], new_coord[1]-vec[1]))
                    coord_shift_left = stateDL.vertexToSite((new_coord[0]+vec[0], new_coord[1]+vec[1]))
                    P2 = view(P[(i,new_coord,direction)], (env.chi,stateDL.site(coord_shift_right).size()[1],env.chi))
                    Pt2 = view(Pt[(i,new_coord,direction)], (env.chi,stateDL.site(new_coord).size()[3],env.chi))
                    P1 = view(P[(i,coord_shift_left,direction)], (env.chi,stateDL.site(new_coord).size()[1],env.chi))
                    Pt1 = view(Pt[(i,coord_shift_left,direction)], (env.chi,stateDL.site(coord_shift_left).size()[3],env.chi))
                    
                    nT = contract(P1, ctm_env_ex2.T[(new_coord,direction)], ([0],[1]))
                    dimsA = state.site(new_coord).size()
                    Aket = state.site(new_coord)
                    DL = torch.reshape(einsum('mefgh,mabcd->eafbgchd',Aket,conj(state.site(new_coord))),\
                              (dimsA[1]**2, dimsA[2]**2, dimsA[3]**2, dimsA[4]**2))
                    nT = contract(nT, DL,([0,2],[1,2]))
                    nT = contract(nT, Pt2,([1,3],[0,1]))
                    tempT2 = contiguous(permute(nT, (1,0,2)))
                    tempT2 = tempT2/tempT2.abs().max()
                    
                    vec_coord = (vec_coord[0],vec_coord[1]-1)
                    new_coord = state.vertexToSite((coord[0]+vec_coord[0], coord[1]+vec_coord[1]))
                    
                    ctm_env_ex2.T[(new_coord,direction)] = tempT2.clone()

                    ################corner tensor################
                    vec_coord_l = (-args.size,args.size-i+1)
                    new_coord_l = state.vertexToSite((coord[0]+vec_coord_l[0], coord[1]+vec_coord_l[1]))
                    coord_shift_right= stateDL.vertexToSite((new_coord_l[0]-vec[0], new_coord_l[1]-vec[1]))
                    coord_shift_left = stateDL.vertexToSite((new_coord_l[0]+vec[0], new_coord_l[1]+vec[1]))
                    P2 = view(P[(i,new_coord_l,direction)], (env.chi,stateDL.site(coord_shift_right).size()[1],env.chi))
                    Pt2 = view(Pt[(i,new_coord_l,direction)], (env.chi,stateDL.site(new_coord_l).size()[3],env.chi))
                    P1 = view(P[(i,coord_shift_left,direction)], (env.chi,stateDL.site(new_coord_l).size()[1],env.chi))
                    Pt1 = view(Pt[(i,coord_shift_left,direction)], (env.chi,stateDL.site(coord_shift_left).size()[3],env.chi))
                    nC1 = contract(ctm_env_ex2.C[(new_coord_l,(-1,1))], ctm_env_ex2.T[(new_coord_l,(-1,0))],([0],[1]))
                    nC1 = contract(nC1, Pt1, ([0,2],[0,1]))
                    tempT2 = nC1/nC1.abs().max()

                    vec_coord = (vec_coord_l[0],vec_coord_l[1]-1)
                    new_coord = state.vertexToSite((coord[0]+vec_coord[0], coord[1]+vec_coord[1]))
                    
                    ctm_env_ex2.C[(new_coord,(-1,1))] = tempT2.clone()
                    
                    vec_coord_r = (args.size+1,args.size-i+1)
                    new_coord_r = state.vertexToSite((coord[0]+vec_coord_r[0], coord[1]+vec_coord_r[1]))
                    coord_shift_right= stateDL.vertexToSite((new_coord_r[0]-vec[0], new_coord_r[1]-vec[1]))
                    coord_shift_left = stateDL.vertexToSite((new_coord_r[0]+vec[0], new_coord_r[1]+vec[1]))
                    P2 = view(P[(i,new_coord_r,direction)], (env.chi,stateDL.site(coord_shift_right).size()[1],env.chi))
                    Pt2 = view(Pt[(i,new_coord_r,direction)], (env.chi,stateDL.site(new_coord_r).size()[3],env.chi))
                    P1 = view(P[(i,coord_shift_left,direction)], (env.chi,stateDL.site(new_coord_r).size()[1],env.chi))
                    Pt1 = view(Pt[(i,coord_shift_left,direction)], (env.chi,stateDL.site(coord_shift_left).size()[3],env.chi))
                    nC2 = contract(ctm_env_ex2.C[(new_coord_r,(1,1))], ctm_env_ex2.T[(new_coord_r,(1,0))],([0],[2]))
                    nC2 = contract(nC2, P2, ([0,2],[0,1]))
                    tempT2 = nC2/nC2.abs().max()

                    vec_coord = (vec_coord_r[0],vec_coord_r[1]-1)
                    new_coord = state.vertexToSite((coord[0]+vec_coord[0], coord[1]+vec_coord[1]))
                    
                    ctm_env_ex2.C[(new_coord,(1,1))] = tempT2.clone()

    return P, Pt

def Create_Norm_Env(state, stateDL, stateB, env, P, Pt, lam, kx, ky, args):
    for coord in stateDL.sites.keys():
        C_up = dict()
        T_up = dict()
        T_up2 = dict()
        C_left = dict()
        T_left = dict()
        T_left2 = dict()
        C_down = dict()
        T_down = dict()
        T_down2 = dict()
        C_right = dict()
        T_right = dict()
        T_right2 = dict()
        for direction in cfg.ctm_args.ctm_move_sequence:
            if direction==(0,-1):
                vec = (1,0)
                vec_coord = (-args.size,-args.size)
                new_coord = state.vertexToSite((coord[0]+vec_coord[0], coord[1]+vec_coord[1]))
                C_up["left"] = env.C[(new_coord,(-1,-1))].clone()
                vec_coord = (args.size+1,-args.size)
                new_coord = state.vertexToSite((coord[0]+vec_coord[0], coord[1]+vec_coord[1]))
                C_up["right"] = env.C[(new_coord,(1,-1))].clone()
                for j in range(2*args.size+2):
                    vec_coord = (-args.size+j,-args.size)
                    new_coord = state.vertexToSite((coord[0]+vec_coord[0], coord[1]+vec_coord[1]))
                    T_up[(j)] = env.T[(new_coord,direction)].clone()
                    
                for i in range(args.size+1):
                    vec_coord_l = (-args.size,-args.size+i)
                    new_coord_l = state.vertexToSite((coord[0]+vec_coord_l[0], coord[1]+vec_coord_l[1]))
                    coord_shift_left= stateDL.vertexToSite((new_coord_l[0]-vec[0], new_coord_l[1]-vec[1]))
                    coord_shift_right = stateDL.vertexToSite((new_coord_l[0]+vec[0], new_coord_l[1]+vec[1]))
                    P2 = view(P[(i,new_coord_l,direction)], (env.chi,stateDL.site(coord_shift_left).size()[3],env.chi))
                    Pt2 = view(Pt[(i,new_coord_l,direction)], (env.chi,stateDL.site(new_coord_l).size()[1],env.chi))
                    P1 = view(P[(i,coord_shift_right,direction)], (env.chi,stateDL.site(new_coord_l).size()[3],env.chi))
                    Pt1 = view(Pt[(i,coord_shift_right,direction)], (env.chi,stateDL.site(coord_shift_right).size()[1],env.chi))
                    nC2 = contract(C_up["left"], env.T[(new_coord_l,(-1,0))],([0],[0]))
                    nC2 = contract(nC2, P2,([0,2],[0,1]))
                    C_up["left"] = nC2/nC2.abs().max()
                    
                    vec_coord_r = (args.size+1,-args.size+i)
                    new_coord_r = state.vertexToSite((coord[0]+vec_coord_r[0], coord[1]+vec_coord_r[1]))
                    coord_shift_left= stateDL.vertexToSite((new_coord_r[0]-vec[0], new_coord_r[1]-vec[1]))
                    coord_shift_right = stateDL.vertexToSite((new_coord_r[0]+vec[0], new_coord_r[1]+vec[1]))
                    P2 = view(P[(i,new_coord_r,direction)], (env.chi,stateDL.site(coord_shift_left).size()[3],env.chi))
                    Pt2 = view(Pt[(i,new_coord_r,direction)], (env.chi,stateDL.site(new_coord_r).size()[1],env.chi))
                    P1 = view(P[(i,coord_shift_right,direction)], (env.chi,stateDL.site(new_coord_r).size()[3],env.chi))
                    Pt1 = view(Pt[(i,coord_shift_right,direction)], (env.chi,stateDL.site(coord_shift_right).size()[1],env.chi))
                    nC1 = contract(C_up["right"], env.T[(new_coord_r,(1,0))],([1],[0]))
                    nC1 = contract(Pt1, nC1,([0,1],[0,1]))
                    C_up["right"] = nC1/nC1.abs().max()

                    for j in range(2*args.size+2):
                        vec_coord = (-args.size+j,-args.size+i)
                        new_coord = state.vertexToSite((coord[0]+vec_coord[0], coord[1]+vec_coord[1]))
                        coord_shift_left= stateDL.vertexToSite((new_coord[0]-vec[0], new_coord[1]-vec[1]))
                        coord_shift_right = stateDL.vertexToSite((new_coord[0]+vec[0], new_coord[1]+vec[1]))
                        P2 = view(P[(i,new_coord,direction)], (env.chi,stateDL.site(coord_shift_left).size()[3],env.chi))
                        Pt2 = view(Pt[(i,new_coord,direction)], (env.chi,stateDL.site(new_coord).size()[1],env.chi))
                        P1 = view(P[(i,coord_shift_right,direction)], (env.chi,stateDL.site(new_coord).size()[3],env.chi))
                        Pt1 = view(Pt[(i,coord_shift_right,direction)], (env.chi,stateDL.site(coord_shift_right).size()[1],env.chi))
                        if i == args.size and j == args.size:
                            nT = contract(Pt2, T_up[(j)], ([0],[0]))
                            dimsA = state.site(new_coord).size()
                            nT = view(nT,(dimsA[2],dimsA[2],env.chi,dimsA[1],dimsA[1],env.chi))
                            Aket = state.site(new_coord) + lam * torch.exp(-1j*(kx*vec_coord[0]+ky*vec_coord[1])) * stateB.site(new_coord)
                            nT = contract(nT, Aket,([0,3],[2,1]))
                            nT = contract(nT, view(P1,(env.chi,dimsA[4],dimsA[4],env.chi)),([3,6],[0,1]))
                            tempT = contiguous(permute(nT, (1,3,2,0,4,5,6)))

                            tempT2 = tempT.detach()

                            T_up[(j)] = tempT/tempT2.abs().max()
                        else:
                            nT = contract(Pt2, T_up[(j)], ([0],[0]))
                            dimsA = state.site(new_coord).size()
                            Aket = state.site(new_coord) + lam * torch.exp(-1j*(kx*vec_coord[0]+ky*vec_coord[1])) * stateB.site(new_coord)
                            DL = view(contiguous(einsum('mefgh,mabcd->eafbgchd',Aket,conj(state.site(new_coord)))),\
                                      (dimsA[1]**2, dimsA[2]**2, dimsA[3]**2, dimsA[4]**2))
                            nT = contract(nT, DL,([0,2],[1,0]))
                            nT = contract(nT, P1,([1,3],[0,1]))
                            tempT = contiguous(nT)

                            tempT2 = tempT.detach()

                            T_up[(j)] = tempT/tempT2.abs().max()

            elif direction==(0,1):
                vec = (-1,0)
                vec_coord = (-args.size,args.size+1)
                new_coord = state.vertexToSite((coord[0]+vec_coord[0], coord[1]+vec_coord[1]))
                C_down["left"] = env.C[(new_coord,(-1,1))].clone()
                vec_coord = (args.size+1,args.size+1)
                new_coord = state.vertexToSite((coord[0]+vec_coord[0], coord[1]+vec_coord[1]))
                C_down["right"] = env.C[(new_coord,(1,1))].clone()
                for j in range(2*args.size+2):
                    vec_coord = (-args.size+j,args.size+1)
                    new_coord = state.vertexToSite((coord[0]+vec_coord[0], coord[1]+vec_coord[1]))
                    T_down[(j)] = env.T[(new_coord,direction)].clone()
                    
                for i in range(args.size+1):
                    vec_coord_l = (-args.size,args.size-i+1)
                    new_coord_l = state.vertexToSite((coord[0]+vec_coord_l[0], coord[1]+vec_coord_l[1]))
                    coord_shift_right= stateDL.vertexToSite((new_coord_l[0]-vec[0], new_coord_l[1]-vec[1]))
                    coord_shift_left = stateDL.vertexToSite((new_coord_l[0]+vec[0], new_coord_l[1]+vec[1]))
                    P2 = view(P[(i,new_coord_l,direction)], (env.chi,stateDL.site(coord_shift_right).size()[1],env.chi))
                    Pt2 = view(Pt[(i,new_coord_l,direction)], (env.chi,stateDL.site(new_coord_l).size()[3],env.chi))
                    P1 = view(P[(i,coord_shift_left,direction)], (env.chi,stateDL.site(new_coord_l).size()[1],env.chi))
                    Pt1 = view(Pt[(i,coord_shift_left,direction)], (env.chi,stateDL.site(coord_shift_left).size()[3],env.chi))
                    nC1 = contract(C_down["left"], env.T[(new_coord_l,(-1,0))],([0],[1]))
                    nC1 = contract(nC1, Pt1, ([0,2],[0,1]))
                    C_down["left"] = nC1/nC1.abs().max()
                    
                    vec_coord_r = (args.size+1,args.size-i+1)
                    new_coord_r = state.vertexToSite((coord[0]+vec_coord_r[0], coord[1]+vec_coord_r[1]))
                    coord_shift_right= stateDL.vertexToSite((new_coord_r[0]-vec[0], new_coord_r[1]-vec[1]))
                    coord_shift_left = stateDL.vertexToSite((new_coord_r[0]+vec[0], new_coord_r[1]+vec[1]))
                    P2 = view(P[(i,new_coord_r,direction)], (env.chi,stateDL.site(coord_shift_right).size()[1],env.chi))
                    Pt2 = view(Pt[(i,new_coord_r,direction)], (env.chi,stateDL.site(new_coord_r).size()[3],env.chi))
                    P1 = view(P[(i,coord_shift_left,direction)], (env.chi,stateDL.site(new_coord_r).size()[1],env.chi))
                    Pt1 = view(Pt[(i,coord_shift_left,direction)], (env.chi,stateDL.site(coord_shift_left).size()[3],env.chi))
                    nC2 = contract(C_down["right"], env.T[(new_coord_r,(1,0))],([0],[2]))
                    nC2 = contract(nC2, P2, ([0,2],[0,1]))
                    C_down["right"] = nC2/nC2.abs().max()

                    for j in range(2*args.size+2):
                        vec_coord = (-args.size+j,args.size-i+1)
                        new_coord = state.vertexToSite((coord[0]+vec_coord[0], coord[1]+vec_coord[1]))
                        coord_shift_right= stateDL.vertexToSite((new_coord[0]-vec[0], new_coord[1]-vec[1]))
                        coord_shift_left = stateDL.vertexToSite((new_coord[0]+vec[0], new_coord[1]+vec[1]))
                        P2 = view(P[(i,new_coord,direction)], (env.chi,stateDL.site(coord_shift_right).size()[1],env.chi))
                        Pt2 = view(Pt[(i,new_coord,direction)], (env.chi,stateDL.site(new_coord).size()[3],env.chi))
                        P1 = view(P[(i,coord_shift_left,direction)], (env.chi,stateDL.site(new_coord).size()[1],env.chi))
                        Pt1 = view(Pt[(i,coord_shift_left,direction)], (env.chi,stateDL.site(coord_shift_left).size()[3],env.chi))
                        nT = contract(P1, T_down[(j)], ([0],[1]))
                        dimsA = state.site(new_coord).size()
                        Aket = state.site(new_coord) + lam * torch.exp(-1j*(kx*vec_coord[0]+ky*vec_coord[1])) * stateB.site(new_coord)
                        DL = view(contiguous(einsum('mefgh,mabcd->eafbgchd',Aket,conj(state.site(new_coord)))),\
                                  (dimsA[1]**2, dimsA[2]**2, dimsA[3]**2, dimsA[4]**2))
                        nT = contract(nT, DL,([0,2],[1,2]))
                        nT = contract(nT, Pt2,([1,3],[0,1]))
                        tempT = contiguous(permute(nT, (1,0,2)))

                        tempT2 = tempT.detach()
                            
                        T_down[(j)] = tempT/tempT2.abs().max()

            elif direction==(-1,0):
                vec = (0,-1)
                vec_coord = (-args.size,-args.size)
                new_coord = state.vertexToSite((coord[0]+vec_coord[0], coord[1]+vec_coord[1]))
                C_left["up"] = env.C[(new_coord,(-1,-1))].clone()
                vec_coord = (-args.size,args.size+1)
                new_coord = state.vertexToSite((coord[0]+vec_coord[0], coord[1]+vec_coord[1]))
                C_left["down"] = env.C[(new_coord,(-1,1))].clone()
                for j in range(2*args.size+2):
                    vec_coord = (-args.size,-args.size+j)
                    new_coord = state.vertexToSite((coord[0]+vec_coord[0], coord[1]+vec_coord[1]))
                    T_left[(j)] = env.T[(new_coord,direction)].clone()
                    
                for i in range(args.size+1):
                    vec_coord_u = (-args.size+i,-args.size)
                    new_coord_u = state.vertexToSite((coord[0]+vec_coord_u[0], coord[1]+vec_coord_u[1]))
                    coord_shift_up= stateDL.vertexToSite((new_coord_u[0]+vec[0], new_coord_u[1]+vec[1]))
                    coord_shift_down= stateDL.vertexToSite((new_coord_u[0]-vec[0], new_coord_u[1]-vec[1]))
                    P2 = view(P[(i,new_coord_u,direction)], (env.chi,stateDL.site(coord_shift_down).size()[0],env.chi))
                    Pt2 = view(Pt[(i,new_coord_u,direction)], (env.chi,stateDL.site(new_coord_u).size()[2],env.chi))
                    P1 = view(P[(i,coord_shift_up,direction)], (env.chi,stateDL.site(new_coord_u).size()[0],env.chi))
                    Pt1 = view(Pt[(i,coord_shift_up,direction)], (env.chi,stateDL.site(coord_shift_up).size()[2],env.chi))
                    nC1 = contract(C_left["up"], env.T[(new_coord_u,(0,-1))],([1],[0]))
                    nC1 = contract(Pt1, nC1,([0,1],[0,1]))
                    C_left["up"] = nC1/nC1.abs().max()
                    
                    vec_coord_d = (-args.size+i,args.size+1)
                    new_coord_d = state.vertexToSite((coord[0]+vec_coord_d[0], coord[1]+vec_coord_d[1]))
                    coord_shift_up= stateDL.vertexToSite((new_coord_d[0]+vec[0], new_coord_d[1]+vec[1]))
                    coord_shift_down= stateDL.vertexToSite((new_coord_d[0]-vec[0], new_coord_d[1]-vec[1]))
                    P2 = view(P[(i,new_coord_d,direction)], (env.chi,stateDL.site(coord_shift_down).size()[0],env.chi))
                    Pt2 = view(Pt[(i,new_coord_d,direction)], (env.chi,stateDL.site(new_coord_d).size()[2],env.chi))
                    P1 = view(P[(i,coord_shift_up,direction)], (env.chi,stateDL.site(new_coord_d).size()[0],env.chi))
                    Pt1 = view(Pt[(i,coord_shift_up,direction)], (env.chi,stateDL.site(coord_shift_up).size()[2],env.chi))
                    nC2 = contract(C_left["down"], env.T[(new_coord_d,(0,1))],([1],[1]))
                    nC2 = contract(P2, nC2,([0,1],[0,1]))
                    C_left["down"] = nC2/nC2.abs().max()

                    for j in range(2*args.size+2):
                        vec_coord = (-args.size+i,-args.size+j)
                        new_coord = state.vertexToSite((coord[0]+vec_coord[0], coord[1]+vec_coord[1]))
                        coord_shift_up= stateDL.vertexToSite((new_coord[0]+vec[0], new_coord[1]+vec[1]))
                        coord_shift_down= stateDL.vertexToSite((new_coord[0]-vec[0], new_coord[1]-vec[1]))
                        P2 = view(P[(i,new_coord,direction)], (env.chi,stateDL.site(coord_shift_down).size()[0],env.chi))
                        Pt2 = view(Pt[(i,new_coord,direction)], (env.chi,stateDL.site(new_coord).size()[2],env.chi))
                        P1 = view(P[(i,coord_shift_up,direction)], (env.chi,stateDL.site(new_coord).size()[0],env.chi))
                        Pt1 = view(Pt[(i,coord_shift_up,direction)], (env.chi,stateDL.site(coord_shift_up).size()[2],env.chi))
                        if i == args.size and j == args.size:
                            nT = contract(P1, T_left[(j)],([0],[0]))
                            dimsA = state.site(new_coord).size()
                            nT = view(nT,(dimsA[1],dimsA[1],env.chi,env.chi,dimsA[2],dimsA[2]))
                            Aket = state.site(new_coord) + lam * torch.exp(-1j*(kx*vec_coord[0]+ky*vec_coord[1])) * stateB.site(new_coord)
                            nT = contract(nT, Aket,([0,4],[1,2]))
                            nT = contract(nT, view(Pt2,(env.chi,dimsA[3],dimsA[3],env.chi)),([2,5],[0,1]))
                            tempT = contiguous(permute(nT, (1,3,0,2,5,4,6)))

                            tempT2 = tempT.detach()
                            
                            T_left[(j)] = tempT/tempT2.abs().max()
                        else:
                            nT = contract(P1, T_left[(j)],([0],[0]))
                            dimsA = state.site(new_coord).size()
                            Aket = state.site(new_coord) + lam * torch.exp(-1j*(kx*vec_coord[0]+ky*vec_coord[1])) * stateB.site(new_coord)
                            DL = view(contiguous(einsum('mefgh,mabcd->eafbgchd',Aket,conj(state.site(new_coord)))),\
                                      (dimsA[1]**2, dimsA[2]**2, dimsA[3]**2, dimsA[4]**2))
                            nT = contract(nT, DL,([0,3],[0,1]))
                            nT = contract(nT, Pt2,([1,2],[0,1]))
                            tempT = contiguous(permute(nT, (0,2,1)))

                            tempT2 = tempT.detach()
                            
                            T_left[(j)] = tempT/tempT2.abs().max()

            elif direction==(1,0):
                vec = (0,1)
                vec_coord = (args.size+1,-args.size)
                new_coord = state.vertexToSite((coord[0]+vec_coord[0], coord[1]+vec_coord[1]))
                C_right["up"] = env.C[(new_coord,(1,-1))].clone()
                vec_coord = (args.size+1,args.size+1)
                new_coord = state.vertexToSite((coord[0]+vec_coord[0], coord[1]+vec_coord[1]))
                C_right["down"] = env.C[(new_coord,(1,1))].clone()
                for j in range(2*args.size+2):
                    vec_coord = (args.size+1,-args.size+j)
                    new_coord = state.vertexToSite((coord[0]+vec_coord[0], coord[1]+vec_coord[1]))
                    T_right[(j)] = env.T[(new_coord,direction)].clone()
                    
                for i in range(args.size+1):
                    vec_coord_u = (args.size-i+1,-args.size)
                    new_coord_u = state.vertexToSite((coord[0]+vec_coord_u[0], coord[1]+vec_coord_u[1]))
                    coord_shift_down = stateDL.vertexToSite((new_coord_u[0]+vec[0], new_coord_u[1]+vec[1]))
                    coord_shift_up = stateDL.vertexToSite((new_coord_u[0]-vec[0], new_coord_u[1]-vec[1]))
                    P2 = view(P[(i,new_coord_u,direction)], (env.chi,stateDL.site(coord_shift_up).size()[2],env.chi))
                    Pt2 = view(Pt[(i,new_coord_u,direction)], (env.chi,stateDL.site(new_coord_u).size()[0],env.chi))
                    P1 = view(P[(i,coord_shift_down,direction)], (env.chi,stateDL.site(new_coord_u).size()[2],env.chi))
                    Pt1 = view(Pt[(i,coord_shift_down,direction)], (env.chi,stateDL.site(coord_shift_down).size()[0],env.chi))
                    nC2 = contract(C_right["up"], env.T[(new_coord_u,(0,-1))],([0],[2]))
                    nC2 = contract(nC2, P2,([0,2],[0,1]))
                    C_right["up"] = nC2/nC2.abs().max()
                    
                    vec_coord_d = (args.size-i+1,args.size+1)
                    new_coord_d = state.vertexToSite((coord[0]+vec_coord_d[0], coord[1]+vec_coord_d[1]))
                    coord_shift_down = stateDL.vertexToSite((new_coord_d[0]+vec[0], new_coord_d[1]+vec[1]))
                    coord_shift_up = stateDL.vertexToSite((new_coord_d[0]-vec[0], new_coord_d[1]-vec[1]))
                    P2 = view(P[(i,new_coord_d,direction)], (env.chi,stateDL.site(coord_shift_up).size()[2],env.chi))
                    Pt2 = view(Pt[(i,new_coord_d,direction)], (env.chi,stateDL.site(new_coord_d).size()[0],env.chi))
                    P1 = view(P[(i,coord_shift_down,direction)], (env.chi,stateDL.site(new_coord_d).size()[2],env.chi))
                    Pt1 = view(Pt[(i,coord_shift_down,direction)], (env.chi,stateDL.site(coord_shift_down).size()[0],env.chi))
                    nC1 = contract(C_right["down"], env.T[(new_coord_d,(0,1))],([1],[2]))
                    nC1 = contract(Pt1, nC1,([0,1],[0,1]))
                    C_right["down"] = nC1/nC1.abs().max()

                    for j in range(2*args.size+2):
                        vec_coord = (args.size-i+1,-args.size+j)
                        new_coord = state.vertexToSite((coord[0]+vec_coord[0], coord[1]+vec_coord[1]))
                        coord_shift_down = stateDL.vertexToSite((new_coord[0]+vec[0], new_coord[1]+vec[1]))
                        coord_shift_up = stateDL.vertexToSite((new_coord[0]-vec[0], new_coord[1]-vec[1]))
                        P2 = view(P[(i,new_coord,direction)], (env.chi,stateDL.site(coord_shift_up).size()[2],env.chi))
                        Pt2 = view(Pt[(i,new_coord,direction)], (env.chi,stateDL.site(new_coord).size()[0],env.chi))
                        P1 = view(P[(i,coord_shift_down,direction)], (env.chi,stateDL.site(new_coord).size()[2],env.chi))
                        Pt1 = view(Pt[(i,coord_shift_down,direction)], (env.chi,stateDL.site(coord_shift_down).size()[0],env.chi))
                        nT = contract(Pt2, T_right[(j)],([0],[0]))
                        dimsA = state.site(new_coord).size()
                        Aket = state.site(new_coord) + lam * torch.exp(-1j*(kx*vec_coord[0]+ky*vec_coord[1])) * stateB.site(new_coord)
                        DL = view(contiguous(einsum('mefgh,mabcd->eafbgchd',Aket,conj(state.site(new_coord)))),\
                                  (dimsA[1]**2, dimsA[2]**2, dimsA[3]**2, dimsA[4]**2))
                        nT = contract(nT, DL,([0,2],[0,3]))
                        nT = contract(nT, P1,([1,3],[0,1]))
                        tempT = contiguous(nT)

                        tempT2 = tempT.detach()
                            
                        T_right[(j)] = tempT/tempT2.abs().max()

    return C_up, T_up, C_left, T_left, C_down, T_down, C_right, T_right

def Create_Norm(state, env, C_up, T_up, C_left, T_left, C_down, T_down, C_right, T_right, args):
    Norm = dict()
    Norm2 = dict()
    for coord in state.sites.keys():
        with torch.no_grad():
            FL = contract(C_up["left"],C_down["left"],([0],[0]))
            FL = FL/FL.abs().max()
            FU = contract(C_left["up"],C_right["up"],([1],[0]))
            FU = FU/FU.abs().max()
            
        for i in range(args.size):
            temp = contract(FL,T_up[(i)],([0],[0]))
            FL = contract(temp,T_down[(i)],([0,1],[1,0]))

            FL2 = FL.detach()
                
            FL = FL/FL2.abs().max()

            temp = contract(FU,T_left[(i)],([0],[0]))
            FU = contract(temp,T_right[(i)],([0,2],[0,1]))

            FU2 = FU.detach()
                
            FU = FU/FU2.abs().max()

        with torch.no_grad():
            FR = contract(C_up["right"],C_down["right"],([1],[0]))
            FR = FR/FR.abs().max()
            FD = contract(C_left["down"],C_right["down"],([1],[1]))
            FD = FD/FD.abs().max()
        
        for i in range(args.size+1):
            temp = contract(FR,T_up[(2*args.size+1-i)],([0],[2]))
            FR = contract(temp,T_down[(2*args.size+1-i)],([0,2],[2,0]))

            FR2 = FR.detach()
                
            FR = FR/FR2.abs().max()

            temp = contract(FD,T_left[(2*args.size+1-i)],([0],[1]))
            FD = contract(temp,T_right[(2*args.size+1-i)],([0,2],[2,1]))

            FD2 = FD.detach()
                
            FD = FD/FD2.abs().max()

        dimsA = state.site(coord).size()
        
        H1 = contract(FL,T_up[(args.size)],([0],[0]))
        H1 = contract(H1,view(T_down[(args.size)],(dimsA[3],dimsA[3],env.chi,env.chi)),([0,4],[2,0]))
        H1 = contiguous(permute(contract(H1,FR,([4,6],[0,1])),(0,1,2,4,3)))

        H12 = H1.detach()

        H1 = H1/H12.abs().max()

        H2 = contract(FU,T_left[(args.size)],([0],[0]))
        H2 = contract(H2,view(T_right[(args.size)],(env.chi,dimsA[4],dimsA[4],env.chi)),([0,5],[0,1]))
        H2 = contract(H2,FD,([4,6],[0,1]))

        H22 = H2.detach()

        H2 = H2/H22.abs().max()

        Norm[coord] = H1/2. + H2/2.
        Norm2[coord] = H1.detach()/2. + H2.detach()/2.

    return Norm, Norm2

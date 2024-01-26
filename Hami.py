import context
import time
import torch
import argparse
import config as cfg
from tn_interface import contract, einsum
from tn_interface import conj
from tn_interface import contiguous, view, permute

def Create_Hami_Env(state, stateDL, stateB, env, P, Pt, lam, H, H2, kx, ky, args):
    phys_dim = state.site((0,0)).size()[0]
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
                            nT = view(nT,(dimsA[2],dimsA[2],env.chi,phys_dim,phys_dim,dimsA[1],dimsA[1],env.chi))
                            Aket = state.site(new_coord) + lam * torch.exp(-1j*(kx*vec_coord[0]+ky*vec_coord[1])) * stateB.site(new_coord)
                            nT = contract(nT, Aket,([0,5],[2,1]))
                            nT = contract(nT, view(P1,(env.chi,dimsA[4],dimsA[4],env.chi)),([5,8],[0,1]))
                            if i%2==1:
                                nT = contract(nT, H,([2,3,5],[0,2,1]))
                                tempT = contiguous(permute(nT, (1,6,2,0,3,4,5)))
                            else:
                                nT = contract(nT, H,([2,3],[0,2]))
                                tempT = contiguous(permute(nT, (1,3,7,8,2,0,4,5,6)))

                            norm = tempT.detach()
                            
                            T_up[(j)] = tempT/norm.abs().max()
                        else:
                            if i == 0:
                                nT = contract(Pt2, T_up[(j)], ([0],[0]))
                                dimsA = state.site(new_coord).size()
                                Aket = state.site(new_coord) + lam * torch.exp(-1j*(kx*vec_coord[0]+ky*vec_coord[1])) * stateB.site(new_coord)
                                DL = view(contiguous(einsum('mefgh,nabcd->mneafbgchd',Aket,conj(state.site(new_coord)))),\
                                          (phys_dim, phys_dim, dimsA[1]**2, dimsA[2]**2, dimsA[3]**2, dimsA[4]**2))
                                nT = contract(nT, DL,([0,2],[3,2]))
                                nT = contract(nT, P1,([1,5],[0,1]))
                                tempT = contiguous(nT)

                                norm = tempT.detach()
                            
                                T_up[(j)] = tempT/norm.abs().max()

                            else:
                                nT = contract(Pt2, T_up[(j)], ([0],[0]))
                                dimsA = state.site(new_coord).size()
                                Aket = state.site(new_coord) + lam * torch.exp(-1j*(kx*vec_coord[0]+ky*vec_coord[1])) * stateB.site(new_coord)
                                DL = view(contiguous(einsum('mefgh,nabcd->mneafbgchd',Aket,conj(state.site(new_coord)))),\
                                          (phys_dim, phys_dim, dimsA[1]**2, dimsA[2]**2, dimsA[3]**2, dimsA[4]**2))
                                nT = contract(nT, DL,([0,4],[3,2]))
                                nT = contract(nT, P1,([3,7],[0,1]))
                                if i%2==1:
                                    nT = contract(nT, H,([1,2,3],[0,2,1]))
                                    tempT = contiguous(permute(nT, (0,4,1,2,3)))
                                else:
                                    nT = contract(nT, H,([1,2,4],[0,2,3]))
                                    tempT = contiguous(permute(nT, (0,1,4,2,3)))

                                norm = tempT.detach()
                            
                                T_up[(j)] = tempT/norm.abs().max()
                                
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
                        if i == 0:
                            nT = contract(P1, T_down[(j)], ([0],[1]))
                            dimsA = state.site(new_coord).size()
                            Aket = state.site(new_coord) + lam * torch.exp(-1j*(kx*vec_coord[0]+ky*vec_coord[1])) * stateB.site(new_coord)
                            DL = view(contiguous(einsum('mefgh,nabcd->mneafbgchd',Aket,conj(state.site(new_coord)))),\
                                      (phys_dim, phys_dim, dimsA[1]**2, dimsA[2]**2, dimsA[3]**2, dimsA[4]**2))
                            nT = contract(nT, DL,([0,2],[3,4]))
                            nT = contract(nT, Pt2,([1,5],[0,1]))
                            tempT = contiguous(nT)#contiguous(permute(nT, (1,0,2)))

                            norm = tempT.detach()
                            
                            T_down[(j)] = tempT/norm.abs().max()
                        else:
                            nT = contract(P1, T_down[(j)], ([0],[0]))
                            dimsA = state.site(new_coord).size()
                            Aket = state.site(new_coord) + lam * torch.exp(-1j*(kx*vec_coord[0]+ky*vec_coord[1])) * stateB.site(new_coord)
                            DL = view(contiguous(einsum('mefgh,nabcd->mneafbgchd',Aket,conj(state.site(new_coord)))),\
                                      (phys_dim, phys_dim, dimsA[1]**2, dimsA[2]**2, dimsA[3]**2, dimsA[4]**2))
                            nT = contract(nT, DL,([0,4],[3,4]))
                            nT = contract(nT, Pt2,([3,7],[0,1]))
                            if i%2==1:
                                nT = contract(nT, H,([1,2,3],[0,2,1]))
                                tempT = contiguous(permute(nT,(0,4,1,2,3)))
                            else:
                                nT = contract(nT, H,([1,2,4],[0,2,3]))
                                tempT = contiguous(permute(nT,(0,1,4,2,3)))

                            norm = tempT.detach()
                            
                            T_down[(j)] = tempT/norm.abs().max()

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
                            nT = view(nT,(dimsA[1],dimsA[1],env.chi,phys_dim,phys_dim,dimsA[2],dimsA[2],env.chi))
                            Aket = state.site(new_coord) + lam * torch.exp(-1j*(kx*vec_coord[0]+ky*vec_coord[1])) * stateB.site(new_coord)
                            nT = contract(nT, Aket,([0,5],[1,2]))
                            nT = contract(nT, view(Pt2,(env.chi,dimsA[3],dimsA[3],env.chi)),([5,7],[0,1]))
                            if i%2==1:
                                nT = contract(nT, H,([2,3,5],[0,2,1]))
                                tempT = contiguous(permute(nT, (1,6,0,2,4,3,5)))
                            else:
                                nT = contract(nT, H,([2,3],[0,2]))
                                tempT = contiguous(permute(nT, (1,3,7,8,0,2,5,4,6)))

                            norm = tempT.detach()
                            
                            T_left[(j)] = tempT/norm.abs().max()
                        else:
                            if i == 0:
                                nT = contract(P1, T_left[(j)],([0],[0]))
                                dimsA = state.site(new_coord).size()
                                Aket = state.site(new_coord) + lam * torch.exp(-1j*(kx*vec_coord[0]+ky*vec_coord[1])) * stateB.site(new_coord)
                                DL = view(contiguous(einsum('mefgh,nabcd->mneafbgchd',Aket,conj(state.site(new_coord)))),\
                                          (phys_dim, phys_dim, dimsA[1]**2, dimsA[2]**2, dimsA[3]**2, dimsA[4]**2))
                                nT = contract(nT, DL,([0,3],[2,3]))
                                nT = contract(nT, Pt2,([1,4],[0,1]))
                                tempT = contiguous(nT)#contiguous(permute(nT, (0,2,1)))

                                norm = tempT.detach()
                            
                                T_left[(j)] = tempT/norm.abs().max()
                            else:
                                nT = contract(P1, T_left[(j)],([0],[0]))
                                dimsA = state.site(new_coord).size()
                                Aket = state.site(new_coord) + lam * torch.exp(-1j*(kx*vec_coord[0]+ky*vec_coord[1])) * stateB.site(new_coord)
                                DL = view(contiguous(einsum('mefgh,nabcd->mneafbgchd',Aket,conj(state.site(new_coord)))),\
                                          (phys_dim, phys_dim, dimsA[1]**2, dimsA[2]**2, dimsA[3]**2, dimsA[4]**2))
                                nT = contract(nT, DL,([0,4],[2,3]))
                                nT = contract(nT, Pt2,([3,6],[0,1]))
                                if i%2==1:
                                    nT = contract(nT, H,([1,2,3],[0,2,1]))
                                    tempT = contiguous(permute(nT,(0,4,1,2,3)))
                                else:
                                    nT = contract(nT, H,([1,2,4],[0,2,3]))
                                    tempT = contiguous(permute(nT,(0,1,4,2,3)))

                                norm = tempT.detach()
                            
                                T_left[(j)] = tempT/norm.abs().max()

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
                        if i == 0:
                            nT = contract(Pt2, T_right[(j)],([0],[0]))
                            dimsA = state.site(new_coord).size()
                            Aket = state.site(new_coord) + lam * torch.exp(-1j*(kx*vec_coord[0]+ky*vec_coord[1])) * stateB.site(new_coord)
                            DL = view(contiguous(einsum('mefgh,nabcd->mneafbgchd',Aket,conj(state.site(new_coord)))),\
                                      (phys_dim, phys_dim, dimsA[1]**2, dimsA[2]**2, dimsA[3]**2, dimsA[4]**2))
                            nT = contract(nT, DL,([0,2],[2,5]))
                            nT = contract(nT, P1,([1,5],[0,1]))
                            tempT = contiguous(nT)

                            norm = tempT.detach()
                            
                            T_right[(j)] = tempT/norm.abs().max()
                        else:
                            nT = contract(Pt2, T_right[(j)],([0],[0]))
                            dimsA = state.site(new_coord).size()
                            Aket = state.site(new_coord) + lam * torch.exp(-1j*(kx*vec_coord[0]+ky*vec_coord[1])) * stateB.site(new_coord)
                            DL = view(contiguous(einsum('mefgh,nabcd->mneafbgchd',Aket,conj(state.site(new_coord)))),\
                                      (phys_dim, phys_dim, dimsA[1]**2, dimsA[2]**2, dimsA[3]**2, dimsA[4]**2))
                            nT = contract(nT, DL,([0,4],[2,5]))
                            nT = contract(nT, P1,([3,7],[0,1]))
                            if i%2==1:
                                nT = contract(nT, H,([1,2,3],[0,2,1]))
                                tempT = contiguous(permute(nT,(0,4,1,2,3)))
                            else:
                                nT = contract(nT, H,([1,2,4],[0,2,3]))
                                tempT = contiguous(permute(nT,(0,1,4,2,3)))

                            norm = tempT.detach()
                            
                            T_right[(j)] = tempT/norm.abs().max()

    return C_up, T_up, C_left, T_left, C_down, T_down, C_right, T_right

def Create_Hami(state, env, C_up, T_up, C_left, T_left, C_down, T_down, C_right, T_right, H, HH, args):
    phys_dim = state.site((0,0)).size()[0]
    Hami = dict()
    Hami2 = dict()
    for coord in state.sites.keys():
        with torch.no_grad():
            FL = contract(C_up["left"],C_down["left"],([0],[0]))
            FL = FL/FL.abs().max()
            FU = contract(C_left["up"],C_right["up"],([1],[0]))
            FU = FU/FU.abs().max()
        
        for i in range(args.size):
            temp = contract(FL,T_up[(i)],([0],[0]))
            temp = contract(temp,T_down[(i)],([0,3],[0,3]))
            FL = contract(temp,H,([0,1,3,4],[0,2,1,3]))

            FL2 = FL.detach()
                
            FL = FL/FL2.abs().max()

            temp = contract(FU,T_left[(i)],([0],[0]))
            temp = contract(temp,T_right[(i)],([0,3],[0,3]))
            FU = contract(temp,H,([0,1,3,4],[0,2,1,3]))

            FU2 = FU.detach()
                
            FU = FU/FU2.abs().max()

        with torch.no_grad():
            FR = contract(C_up["right"],C_down["right"],([1],[0]))
            FR = FR/FR.abs().max()
            FD = contract(C_left["down"],C_right["down"],([1],[1]))
            FD = FD/FD.abs().max()
        
        for i in range(args.size+1):
            temp = contract(FR,T_up[(2*args.size+1-i)],([0],[4]))
            temp = contract(temp,T_down[(2*args.size+1-i)],([0,4],[4,3]))
            FR = contract(temp,H,([1,2,4,5],[0,2,1,3]))

            FR2 = FR.detach()
                
            FR = FR/FR2.abs().max()

            temp = contract(FD,T_left[(2*args.size+1-i)],([0],[4]))
            temp = contract(temp,T_right[(2*args.size+1-i)],([0,4],[4,3]))
            FD = contract(temp,H,([1,2,4,5],[0,2,1,3]))

            FD2 = FD.detach()
                
            FD = FD/FD2.abs().max()

        dimsA = state.site(coord).size()

        if args.size%2==1:
            H1 = contract(FL,T_up[(args.size)],([0],[0]))
            H1 = contract(H1,view(T_down[(args.size)],(env.chi,phys_dim,phys_dim,dimsA[3],dimsA[3],env.chi)),([0,4],[0,3]))
            H1 = contract(H1,H,([0,5,6],[0,1,3]))
            H1 = contiguous(permute(contract(H1,FR,([3,5],[0,1])),(4,0,1,3,2)))

            H12 = H1.detach()

            H1 = H1/H12.abs().max()

            H2 = contract(FU,T_left[(args.size)],([0],[0]))
            H2 = contract(H2,view(T_right[(args.size)],(env.chi,phys_dim,phys_dim,dimsA[4],dimsA[4],env.chi)),([0,5],[0,3]))
            H2 = contract(H2,H,([0,5,6],[0,1,3]))
            H2 = contiguous(permute(contract(H2,FD,([3,5],[0,1])),(4,0,1,2,3)))

            H22 = H2.detach()

            H2 = H2/H22.abs().max()
        else:
            H1 = contract(FL,T_up[(args.size)],([0],[0]))
            H1 = contract(H1,view(T_down[(args.size)],(env.chi,phys_dim,phys_dim,dimsA[3],dimsA[3],env.chi)),([0,6],[0,3]))
            H1 = contract(H1,H,([0,1,7,8],[0,2,1,3]))
            H1 = contiguous(permute(contract(H1,FR,([4,6],[0,1])),(0,1,2,4,3)))

            H12 = H1.detach()

            H1 = H1/H12.abs().max()

            H2 = contract(FU,T_left[(args.size)],([0],[0]))
            H2 = contract(H2,view(T_right[(args.size)],(env.chi,phys_dim,phys_dim,dimsA[4],dimsA[4],env.chi)),([0,7],[0,3]))
            H2 = contract(H2,H,([0,1,7,8],[0,2,1,3]))
            H2 = contract(H2,FD,([4,6],[0,1]))

            H22 = H2.detach()

            H2 = H2/H22.abs().max()

        Hami[coord] = H1/2. + H2/2.
        Hami2[coord] = H1.detach()/2. + H2.detach()/2.

    return Hami, Hami2

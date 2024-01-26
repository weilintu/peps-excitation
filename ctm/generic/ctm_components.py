import torch
from torch.utils.checkpoint import checkpoint
from config import ctm_args
from tn_interface import contract
from tn_interface import view, permute, contiguous

#####################################################################
# functions building pair of 4x2 (or 2x4) halves of 4x4 TN
#####################################################################
def halves_of_4x4_CTM_MOVE_UP(coord, state, env, verbosity=0):
    # C T T        C = C2x2_LU(coord+(-1,0))  C2x2(coord)
    # T A B(coord) T   C2x2_LD(coord+(-1,-1)) C2x2(coord+(0,1))
    # T C D        T
    # C T T        C

    # C2x2--1->0 0--C2x2(coord) =     _0 0_
    # |0           1|                |     |
    # |0           0|             half2    half1
    # C2x2--1    1--C2x2             |_1 1_|
    
    # RU, RD, LU, LD
    tensors= c2x2_RU_t(coord,state,env) + c2x2_RD_t((coord[0], coord[1]+1),state,env) \
        + c2x2_LU_t((coord[0]-1, coord[1]),state,env) + c2x2_LD_t((coord[0]-1, coord[1]-1),state,env)

    if ctm_args.fwd_checkpoint_halves:
        return checkpoint(halves_of_4x4_CTM_MOVE_UP_c,*tensors)
    else:
        return halves_of_4x4_CTM_MOVE_UP_c(*tensors)

def halves_of_4x4_CTM_MOVE_UP_t(coord, state, env):
    # RU, RD, LU, LD
    tensors= c2x2_RU_t(coord,state,env) + c2x2_RD_t((coord[0], coord[1]+1),state,env) \
        + c2x2_LU_t((coord[0]-1, coord[1]),state,env) + c2x2_LD_t((coord[0]-1, coord[1]-1),state,env)
    return tensors

def halves_of_4x4_CTM_MOVE_UP_c(*tensors):
    # C T T        C = C2x2_LU(coord+(-1,0))  C2x2(coord)
    # T A B(coord) T   C2x2_LD(coord+(-1,-1)) C2x2(coord+(0,1))
    # T C D        T
    # C T T        C

    # C2x2--1->0 0--C2x2(coord) =     _0 0_
    # |0           1|                |     |
    # |0           0|             half2    half1
    # C2x2--1    1--C2x2             |_1 1_|
    
    # C_1, T1_1, T2_1, A_1= tensors[0:4]
    # C_2, T1_2, T2_2, A_2= tensors[4:8]
    # C_3, T1_3, T2_3, A_3= tensors[8:12]
    # C_4, T1_4, T2_4, A_4= tensors[12:16]

    return contract(c2x2_RU_c(*tensors[0:4]),c2x2_RD_c(*tensors[4:8]),([1],[0])), \
        contract(c2x2_LU_c(*tensors[8:12]),c2x2_LD_c(*tensors[12:16]),([0],[0]))

def halves_of_4x4_CTM_MOVE_LEFT(coord, state, env, verbosity=0):
    # C T        T C = C2x2_LU(coord)       C2x2(coord+(1,0))
    # T A(coord) B T   C2x2_LD(coord+(0,1)) C2x2(coord+(1,1))
    # T C        D T
    # C T        T C

    # C2x2(coord)--1 0--C2x2 = half1
    # |0               1|      |0  |1
    # 
    # |0            1<-0|      |0  |1
    # C2x2--1 1---------C2x2   half2
    
    # LU, RU, LS, RD
    tensors= c2x2_LU_t(coord,state,env) + c2x2_RU_t((coord[0]+1, coord[1]),state,env) \
        + c2x2_LD_t((coord[0], coord[1]+1),state,env) + c2x2_RD_t((coord[0]+1, coord[1]+1),state,env)

    if ctm_args.fwd_checkpoint_halves:
        return checkpoint(halves_of_4x4_CTM_MOVE_LEFT_c,*tensors)
    else:
        return halves_of_4x4_CTM_MOVE_LEFT_c(*tensors)

def halves_of_4x4_CTM_MOVE_LEFT_t(coord, state, env):
    # LU, RU, LS, RD
    tensors= c2x2_LU_t(coord,state,env) + c2x2_RU_t((coord[0]+1, coord[1]),state,env) \
        + c2x2_LD_t((coord[0], coord[1]+1),state,env) + c2x2_RD_t((coord[0]+1, coord[1]+1),state,env)
    return tensors

def halves_of_4x4_CTM_MOVE_LEFT_c(*tensors):
    # C T        T C = C2x2_LU(coord)       C2x2(coord+(1,0))
    # T A(coord) B T   C2x2_LD(coord+(0,1)) C2x2(coord+(1,1))
    # T C        D T
    # C T        T C

    # C2x2(coord)--1 0--C2x2 = half1
    # |0               1|      |0  |1
    # 
    # |0            1<-0|      |0  |1
    # C2x2--1 1---------C2x2   half2
    
    return contract(c2x2_LU_c(*tensors[0:4]),c2x2_RU_c(*tensors[4:8]),([1],[0])), \
        contract(c2x2_LD_c(*tensors[8:12]),c2x2_RD_c(*tensors[12:16]),([1],[1]))

def halves_of_4x4_CTM_MOVE_DOWN(coord, state, env, verbosity=0):
    # C T T        C = C2x2_LU(coord+(0,-1)) C2x2(coord+(1,-1))
    # T A        B T   C2x2_LD(coord)        C2x2(coord+(1,0))
    # T C(coord) D T
    # C T        T C

    # C2x2---------1    1<-0--C2x2 =     _1 1_
    # |0                      |1        |     |
    # |0                      |0      half1    half2
    # C2x2(coord)--1->0 0<-1--C2x2      |_0 0_|

    # LD, LU, RD, RU
    tensors= c2x2_LD_t(coord,state,env) + c2x2_LU_t((coord[0], coord[1]-1),state,env) \
        + c2x2_RD_t((coord[0]+1, coord[1]),state,env) + c2x2_RU_t((coord[0]+1, coord[1]-1),state,env)
    
    if ctm_args.fwd_checkpoint_halves:
        return checkpoint(halves_of_4x4_CTM_MOVE_DOWN_c,*tensors)
    else:
        return halves_of_4x4_CTM_MOVE_DOWN_c(*tensors)

def halves_of_4x4_CTM_MOVE_DOWN_t(coord, state, env):
    # LD, LU, RD, RU
    tensors= c2x2_LD_t(coord,state,env) + c2x2_LU_t((coord[0], coord[1]-1),state,env) \
        + c2x2_RD_t((coord[0]+1, coord[1]),state,env) + c2x2_RU_t((coord[0]+1, coord[1]-1),state,env)
    return tensors

def halves_of_4x4_CTM_MOVE_DOWN_c(*tensors):
    # C T T        C = C2x2_LU(coord+(0,-1)) C2x2(coord+(1,-1))
    # T A        B T   C2x2_LD(coord)        C2x2(coord+(1,0))
    # T C(coord) D T
    # C T        T C

    # C2x2---------1    1<-0--C2x2 =     _1 1_
    # |0                      |1        |     |
    # |0                      |0      half1    half2
    # C2x2(coord)--1->0 0<-1--C2x2      |_0 0_|

    return contract(c2x2_LD_c(*tensors[0:4]),c2x2_LU_c(*tensors[4:8]),([0],[0])), \
        contract(c2x2_RD_c(*tensors[8:12]),c2x2_RU_c(*tensors[12:16]),([0],[1]))

def halves_of_4x4_CTM_MOVE_RIGHT(coord, state, env, verbosity=0):
    # C T T        C = C2x2_LU(coord+(-1,-1)) C2x2(coord+(0,-1))
    # T A B        T   C2x2_LD(coord+(-1,0))  C2x2(coord)
    # T C D(coord) T
    # C T T        C

    # C2x2--1 0--C2x2        = half2
    # |0->1      |1->0         |1  |0
    # 
    # |0->1      |0            |1  |0
    # C2x2--1 1--C2x2(coord)   half1
    
    # RD, LD, RU, LU
    tensors= c2x2_RD_t(coord,state,env) + c2x2_LD_t((coord[0]-1, coord[1]),state,env) \
        + c2x2_RU_t((coord[0], coord[1]-1),state,env) + c2x2_LU_t((coord[0]-1, coord[1]-1),state,env)
    
    if ctm_args.fwd_checkpoint_halves:
        return checkpoint(halves_of_4x4_CTM_MOVE_RIGHT_c,*tensors)
    else:
        return halves_of_4x4_CTM_MOVE_RIGHT_c(*tensors)

def halves_of_4x4_CTM_MOVE_RIGHT_t(coord, state, env):
    # RD, LD, RU, LU
    tensors= c2x2_RD_t(coord,state,env) + c2x2_LD_t((coord[0]-1, coord[1]),state,env) \
        + c2x2_RU_t((coord[0], coord[1]-1),state,env) + c2x2_LU_t((coord[0]-1, coord[1]-1),state,env)
    return tensors

def halves_of_4x4_CTM_MOVE_RIGHT_c(*tensors):
    # C T T        C = C2x2_LU(coord+(-1,-1)) C2x2(coord+(0,-1))
    # T A B        T   C2x2_LD(coord+(-1,0))  C2x2(coord)
    # T C D(coord) T
    # C T T        C

    # C2x2--1 0--C2x2        = half2
    # |0->1      |1->0         |1  |0
    # 
    # |0->1      |0            |1  |0
    # C2x2--1 1--C2x2(coord)   half1

    return contract(c2x2_RD_c(*tensors[0:4]),c2x2_LD_c(*tensors[4:8]),([1],[1])), \
        contract(c2x2_RU_c(*tensors[8:12]),c2x2_LU_c(*tensors[12:16]),([0],[1]))

#####################################################################
# functions building 2x2 Corner
#####################################################################
def c2x2_LU(coord, state, env, verbosity=0):
    C = env.C[(state.vertexToSite(coord),(-1,-1))]
    T1 = env.T[(state.vertexToSite(coord),(0,-1))]
    T2 = env.T[(state.vertexToSite(coord),(-1,0))]
    A = state.site(coord)

    tensors= C, T1, T2, A

    if ctm_args.fwd_checkpoint_c2x2:
        C2x2= checkpoint(c2x2_LU_c,*tensors)
    else:
        C2x2= c2x2_LU_c(*tensors)

    if verbosity>0:
        print("C2X2 LU "+str(coord)+"->"+str(state.vertexToSite(coord))+" (-1,-1)")
    if verbosity>1:
        print(C2x2)

    return C2x2

def c2x2_LU_t(coord, state, env):
    tensors= env.C[(state.vertexToSite(coord),(-1,-1))], \
        env.T[(state.vertexToSite(coord),(0,-1))], \
        env.T[(state.vertexToSite(coord),(-1,0))], \
        state.site(coord)
    return tensors

def c2x2_LU_c(*tensors):
        C, T1, T2, A= tensors
        # C--10--T1--2
        # 0      1
        C2x2 = contract(C, T1, ([1],[0]))

        # C------T1--2->1
        # 0      1->0
        # 0
        # T2--2->3
        # 1->2
        C2x2 = contract(C2x2, T2, ([0],[0]))

        # C-------T1--1->0
        # |       0
        # |       0
        # T2--3 1 A--3 
        # 2->1    2
        C2x2 = contract(C2x2, A, ([0,3],[0,1]))

        # permute 0123->1203
        # reshape (12)(03)->01
        C2x2 = contiguous(permute(C2x2,(1,2,0,3)))
        C2x2 = view(C2x2,(T2.size(1)*A.size(2),T1.size(2)*A.size(3)))

        # C2x2--1
        # |
        # 0
        return C2x2

def c2x2_RU(coord, state, env, verbosity=0):
    C = env.C[(state.vertexToSite(coord),(1,-1))]
    T1 = env.T[(state.vertexToSite(coord),(1,0))]
    T2 = env.T[(state.vertexToSite(coord),(0,-1))]
    A = state.site(coord)

    tensors= C, T1, T2, A

    if ctm_args.fwd_checkpoint_c2x2:
        C2x2= checkpoint(c2x2_RU_c,*tensors)
    else:
        C2x2= c2x2_RU_c(*tensors)

    if verbosity>0:
        print("C2X2 RU "+str(coord)+"->"+str(state.vertexToSite(coord))+" (1,-1)")
    if verbosity>1: 
        print(C2x2)

    return C2x2

def c2x2_RU_t(coord, state, env):
    tensors= env.C[(state.vertexToSite(coord),(1,-1))], \
        env.T[(state.vertexToSite(coord),(1,0))], \
        env.T[(state.vertexToSite(coord),(0,-1))], \
        state.site(coord)
    return tensors

def c2x2_RU_c(*tensors):
        C, T1, T2, A= tensors 
        # 0--C
        #    1
        #    0
        # 1--T1
        #    2
        C2x2 = contract(C, T1, ([1],[0]))

        # 2<-0--T2--2 0--C
        #    3<-1        |
        #          0<-1--T1
        #             1<-2
        C2x2 = contract(C2x2, T2, ([0],[2]))

        # 1<-2--T2------C
        #       3       |
        #       0       |
        # 2<-1--A--3 0--T1
        #    3<-2    0<-1
        C2x2 = contract(C2x2, A, ([0,3],[3,0]))

        # permute 0123->1203
        # reshape (12)(03)->01
        C2x2 = contiguous(permute(C2x2,(1,2,0,3)))
        C2x2 = view(C2x2,(T2.size(0)*A.size(1),T1.size(2)*A.size(2)))
     
        # 0--C2x2
        #    |
        #    1
        return C2x2

def c2x2_RD(coord, state, env, verbosity=0):
    C = env.C[(state.vertexToSite(coord),(1,1))]
    T1 = env.T[(state.vertexToSite(coord),(0,1))]
    T2 = env.T[(state.vertexToSite(coord),(1,0))]
    A = state.site(coord)

    tensors= C, T1, T2, A

    if ctm_args.fwd_checkpoint_c2x2:
        C2x2= checkpoint(c2x2_RD_c,*tensors)
    else:
        C2x2= c2x2_RD_c(*tensors)

    if verbosity>0:
        print("C2X2 RD "+str(coord)+"->"+str(state.vertexToSite(coord))+" (1,1)")
    if verbosity>1:
        print(C2x2)

    return C2x2

def c2x2_RD_t(coord, state, env):
    tensors= env.C[(state.vertexToSite(coord),(1,1))], \
        env.T[(state.vertexToSite(coord),(0,1))], \
        env.T[(state.vertexToSite(coord),(1,0))], \
        state.site(coord)
    return tensors

def c2x2_RD_c(*tensors):
        C, T1, T2, A= tensors
        #    1<-0        0
        # 2<-1--T1--2 1--C
        C2x2 = contract(C, T1, ([1],[2]))

        #         2<-0
        #      3<-1--T2
        #            2
        #    0<-1    0
        # 1<-2--T1---C
        C2x2 = contract(C2x2, T2, ([0],[2]))

        #    2<-0    1<-2
        # 3<-1--A--3 3--T2
        #       2       |
        #       0       |
        # 0<-1--T1------C
        C2x2 = contract(C2x2, A, ([0,3],[2,3]))

        # permute 0123->1203
        # reshape (12)(03)->01
        C2x2 = contiguous(permute(C2x2,(1,2,0,3)))
        C2x2 = view(C2x2,(T2.size(0)*A.size(0),T1.size(1)*A.size(1)))

        #    0
        #    |
        # 1--C2x2
        return C2x2

def c2x2_LD(coord, state, env, verbosity=0):
    C = env.C[(state.vertexToSite(coord),(-1,1))]
    T1 = env.T[(state.vertexToSite(coord),(-1,0))]
    T2 = env.T[(state.vertexToSite(coord),(0,1))]
    A = state.site(coord)

    tensors= C, T1, T2, A

    if ctm_args.fwd_checkpoint_c2x2:
        C2x2= checkpoint(c2x2_LD_c,*tensors)
    else:
        C2x2= c2x2_LD_c(*tensors)

    if verbosity>0: 
        print("C2X2 LD "+str(coord)+"->"+str(state.vertexToSite(coord))+" (-1,1)")
    if verbosity>1:
        print(C2x2)

    return C2x2

def c2x2_LD_t(coord, state, env):
    tensors= env.C[(state.vertexToSite(coord),(-1,1))], \
        env.T[(state.vertexToSite(coord),(-1,0))], \
        env.T[(state.vertexToSite(coord),(0,1))], \
        state.site(coord)
    return tensors

def c2x2_LD_c(*tensors):
        C, T1, T2, A= tensors
        # 0->1
        # T1--2
        # 1
        # 0
        # C--1->0
        C2x2 = contract(C, T1, ([0],[1]))

        # 1->0
        # T1--2->1
        # |
        # |       0->2
        # C--0 1--T1--2->3
        C2x2 = contract(C2x2, T2, ([0],[1]))

        # 0       0->2
        # T1--1 1--A--3
        # |        2
        # |        2
        # C--------T2--3->1
        C2x2 = contract(C2x2, A, ([1,2],[1,2]))

        # permute 0123->0213
        # reshape (02)(13)->01
        C2x2 = contiguous(permute(C2x2,(0,2,1,3)))
        C2x2 = view(C2x2,(T1.size(0)*A.size(0),T2.size(2)*A.size(3)))

        # 0
        # |
        # C2x2--1
        return C2x2

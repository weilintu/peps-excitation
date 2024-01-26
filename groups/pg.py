import torch

def make_d2_symm(A):
    r"""
    :param A: on-site tensor
    :type A: torch.tensor
    :return: d2 symmetrized tensor ``A``
    :rtype: torch.tensor

    ::
           u s 
           |/ 
        l--A--r  <=> A[s,u,l,d,r]
           |
           d
    
    Perform left-right symmetrization
    """
    A= 0.5*(A + A.permute(0,1,4,3,2))   # left-right symmetry
    return A

def make_d2_antisymm(A):
    r"""
    :param A: on-site tensor
    :type A: torch.tensor
    :return: d2 anti-symmetrized tensor ``A``
    :rtype: torch.tensor

    ::
           u s 
           |/ 
        l--A--r  <=> A[s,u,l,d,r]
           |
           d
    
    Perform left-right symmetrization
    """
    A= 0.5*(A - A.permute(0,1,4,3,2))   # left-right symmetry
    return A

def make_c4v_symm(A, irreps=["A1"]):
    r"""
    :param A: on-site tensor
    :param irreps: choice of irreps from A1, A2, B1, or B2
    :type A: torch.tensor
    :type irreps: list(str)
    :return: C4v symmetrized tensor ``A``
    :rtype: torch.tensor

    ::
           u s 
           |/ 
        l--A--r  <=> A[s,u,l,d,r]
           |
           d
    
    Project and sum any combination of projections on real C4v irreps A1, A2, B1, 
    and B2. List of irreps is converted to a set (no repeated elements) and the 
    projections are then summed up.
    """
    projections=dict({"A1": make_c4v_symm_A1, "A2": make_c4v_symm_A2, \
      "B1": make_c4v_symm_B1, "B2": make_c4v_symm_B2})
    irreps=set(irreps)
    assert irreps.issubset(set(projections.keys())), "Unknown C4v irrep"
    A_symm= torch.zeros(A.size(),device=A.device,dtype=A.dtype)
    for irrep in irreps:
      A_symm= A_symm + projections[irrep](A)
    return A_symm

def make_c4v_symm_A1(A):
    r"""
    :param A: on-site tensor
    :type A: torch.tensor
    :return: c4v symmetrized tensor ``A``
    :rtype: torch.tensor

    ::
           u s 
           |/ 
        l--A--r  <=> A[s,u,l,d,r]
           |
           d
    
    Project on-site tensor ``A`` on A1 irrep of C4v group.
    """
    A= 0.5*(A + A.permute(0,1,4,3,2))   # left-right reflection
    A= 0.5*(A + A.permute(0,3,2,1,4))   # up-down reflection
    A= 0.5*(A + A.permute(0,4,1,2,3))   # pi/2 anti-clockwise
    A= 0.5*(A + A.permute(0,2,3,4,1))   # pi/2 clockwise

    return A

def make_c4v_symm_A2(A):
    r"""
    :param A: on-site tensor
    :type A: torch.tensor
    :return: c4v symmetrized tensor ``A``
    :rtype: torch.tensor

    ::
           u s 
           |/ 
        l--A--r  <=> A[s,u,l,d,r]
           |
           d
    
    Project on-site tensor ``A`` on A2 irrep of C4v group.
    """
    A= 0.5*(A - A.permute(0,1,4,3,2))   # left-right reflection (\sigma)
    A= 0.5*(A - A.permute(0,4,3,2,1))   # skew reflection (\sigma R^-1) 
    A= 0.5*(A + A.permute(0,4,1,2,3))   # pi/2 anti-clockwise (R)
    A= 0.5*(A + A.permute(0,3,4,1,2))   # pi anti-clockwise (R^2)
    return A

def make_c4v_symm_B1(A):
    r"""
    :param A: on-site tensor
    :type A: torch.tensor
    :return: c4v symmetrized tensor ``A``
    :rtype: torch.tensor

    ::
           u s 
           |/ 
        l--A--r  <=> A[s,u,l,d,r]
           |
           d
    
    Project on-site tensor ``A`` on B1 irrep of C4v group.
    """
    A= 0.5*(A + A.permute(0,1,4,3,2))   # left-right reflection (\sigma)
    A= 0.5*(A - A.permute(0,4,3,2,1))   # skew reflection (\sigma R^-1) 
    A= 0.5*(A - A.permute(0,4,1,2,3))   # pi/2 anti-clockwise (R)
    A= 0.5*(A + A.permute(0,3,4,1,2))   # pi anti-clockwise (R^2)
    return A

def make_c4v_symm_B2(A):
    r"""
    :param A: on-site tensor
    :type A: torch.tensor
    :return: C4v symmetrized tensor ``A``
    :rtype: torch.tensor

    ::
           u s 
           |/ 
        l--A--r  <=> A[s,u,l,d,r]
           |
           d
    
    Project on-site tensor ``A`` on B2 irrep of C4v group.
    """
    A= 0.5*(A - A.permute(0,1,4,3,2))   # left-right reflection (\sigma)
    A= 0.5*(A + A.permute(0,4,3,2,1))   # skew reflection (\sigma R^-1) 
    A= 0.5*(A + A.permute(0,4,1,2,3))   # pi/2 anti-clockwise (R)
    A= 0.5*(A - A.permute(0,3,4,1,2))   # pi anti-clockwise (R^2)
    return A

def verify_c4v_symm_A1(A):
    with torch.no_grad():
        symm= True
        max_d=0.
        d_list=[]
        for p in [(0,1,4,3,2), (0,3,2,1,4), (0,4,1,2,3), (0,2,3,4,1)]:
            d= torch.dist(A,A.permute(p))
            d_list.append((p,d))
            symm= symm * (d<tol)
            max_d= max(max_d,d)
        return symm, max_d, d_list

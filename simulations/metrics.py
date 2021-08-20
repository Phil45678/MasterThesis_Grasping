# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 16:27:47 2020

@author: norman marlier

Grasping metrics



Inspired by:
https://github.com/BerkeleyAutomation/dex-net/blob/cccf93319095374b0eefc24b8b6cd40bc23966d2/src/dexnet/grasping/quality.py#L235
"""
import numpy as np
from scipy.optimize import linprog
from scipy.spatial import ConvexHull


def min_singular(G):
    """Minimum singular value of G.

    Roa, M. A., & Suárez, R. (2015).
    Grasp quality measures: review and performance.
    Autonomous robots, 38(1), 65-88.

    Parameters
    ----------
    G: an numpy array (6x3N), the grasp matrix

    Return
    ------
    Qmsv: a float, the min singular value of G
    """
    _, S, _ = np.linalg.svd(G)
    if S.size != 6:
        return 0
    else:
        return S[-1]


def wrench_volume(G):
    """Volume ofthe ellipsoid in the wrench space.

    Roa, M. A., & Suárez, R. (2015).
    Grasp quality measures: review and performance.
    Autonomous robots, 38(1), 65-88.

    Parameters
    ----------
    -G: an numpy array (6x3N), the grasp matrix

    Return
    ------
    -Qvew: a float, the volume of the ellipsoid in the wrench space
    """
    _, S, _ = np.linalg.svd(G)
    if S.size != 6:
        return 0
    else:
        return np.prod(S)


def grasp_isotropy(G):
    """Grasp isotropy index.

    Roa, M. A., & Suárez, R. (2015).
    Grasp quality measures: review and performance.
    Autonomous robots, 38(1), 65-88.

    Parameters
    ----------
    -G: an numpy array (6x3N), the grasp matrix

    Return
    ------
    -Qgii: a float [0, 1], the relation between min_singular(Q) and
    max_singular(Q)
    """
    _, S, _ = np.linalg.svd(G)
    if S.size != 6:
        return 0
    else:
        return S[-1]/S[0]


def form_closure(F):
    """Form closure of a grasp.
    
    Lynch, K. M., & Park, F. C. (2017).
    Modern Robotics.
    Cambridge University Press.
    
    min k
    subject to 
        Fk = 0
        k >= 1
    
    Parameters
    ----------
    -F: an numpy array (6xN), the grasp map
    
    Return
    ------
    -form_closure: a bool, True if it is form-closure, False otherwise
    """
    # Dimension
    dim = F.shape
    # Full rank
    cond1 = np.linalg.matrix_rank(F) == dim[0]
    if cond1:
        # Form closure ?
        # Coefficient
        c = np.ones((dim[1], 1), dtype=np.float32)
        # Equality constraints
        Aeq = F
        beq = np.zeros((dim[0], 1), dtype=np.float32)
        # Lin prog
        res = linprog(c=c, A_eq=Aeq, b_eq=beq, bounds=(1., None))
        cond2 = res.success
        # Form closure ?
        return int(cond1 & cond2)
    else:
        return 0.


def force_closure(W):
    """Force closure of a grasp.
    
    Origin is inside the convex hull of W
    
    Parameters
    ----------
    -W: an numpy array (6xNg), the  grasp wrench space
    
    Return
    ------
    -form_closure: a bool, True if it is form-closure, False otherwise
    """
    if W.shape[1] == 0.:
        return 0.
    else:
        try:
            CH_W = ConvexHull(W.T)
            CH_W.close()
            return CH_W.volume > 0.
        except:
            print("Cannot create a convex hull")
            return 0.

def VolumeCH(W):
    """Force closure of a grasp.
    
    Origin is inside the convex hull of W
    
    Parameters
    ----------
    -W: an numpy array (6xNg), the  grasp wrench space
    
    Return
    ------
    -form_closure: a bool, True if it is form-closure, False otherwise
    """
    if W.shape[1] == 0.:
        return 0.
    else:
        try:
            CH_W = ConvexHull(W.T)
            CH_W.close()
            return CH_W.volume
        except:
            return 0.
        



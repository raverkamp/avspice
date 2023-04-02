"""utility methods"""

import math
import sys

from typing import Optional, Any, Callable,Union

import numpy as np
import numpy.typing as npt

def drange(start:float, end:float, step:Optional[float]=None)->list[float]:
    x = float(start)
    l = []
    if step is None:
        s = 1.0
    else:
        s = step
    s = float(s)
    if s <=0:
        raise Exception("step <=0")
    while x < end:
        l.append(x)
        x += s
    if x < end + s/2.0:
        l.append(end)
    return l

def saw_tooth(freq:float, t:float) ->  float:
    t = 1.0 * t * freq
    t = t % 1
    if t<0.5:
        return t*2
    else:
        return 2 - t*2


def explin(x: float, lcutoff: float, rcutoff:float) ->float:
    assert lcutoff  <= rcutoff, "cutoffs wrong"

    if x > rcutoff:
        return math.exp(rcutoff) +  (x-rcutoff) * math.exp(rcutoff)
    elif x < lcutoff:
        return math.exp(lcutoff) +  (x-lcutoff) * math.exp(lcutoff)
    else:
        return math.exp(x)

def dexplin(x:float, lcutoff:float, rcutoff:float) -> float:
    assert lcutoff  <= rcutoff, "cutoffs wrong"

    if x > rcutoff:
        return math.exp(rcutoff)
    elif x < lcutoff:
        return  math.exp(lcutoff)
    else:
        return math.exp(x)

def is_str_seq(l:Any) -> bool:
    for x  in l:
        if not isinstance(x, str):
            return False
    return True


def find_pos(v: Union[npt.NDArray[np.float64],list[float]], x:float) -> int:
    """v is sorted double vector, x is double
       return largest i such that  x >= v[i], -1 if x< v[0]"""
    if len(v) == 0:
        raise Exception("argument v must have length >=1")
    if x < v[0]:
        return -1
    i = len(v) -1
    while True:
        if i == -1:
            return -1
        if x >= v[i]:
            return i
        i-=1

def linear_interpolate(x: Union[npt.NDArray[np.float64],list[float]], y:Union[npt.NDArray[np.float64],list[float]], t:float) -> float:
    i = find_pos(x,t)
    if i == -1:
        return y[0]
    if i == len(x) -1:
        return y[i]
    ax = x[i]
    bx = x[i+1]
    ay = y[i]
    by = y[i+1]
    return ay + (t-ax)/ (bx-ax) * (by-ay)

def smooth_step(left:float, right:float, x:float) -> float:
    x = (x- left) /  (right - left)
    if x<=0:
        return 0.0
    if x>=1:
        return 1.0
    return 3 * x * x  - 2 * x * x * x

def dsmooth_step(left:float, right:float, x:float) -> float:
    x = (x- left) /   (right - left)
    if x<=0:
        return 0.0
    if x>=1:
        return 0.0
    return (6 * x - 6 * x * x) /  (right - left)

def eps() -> float:
    return sys.float_info.epsilon

def ndiff(fun:Callable[[float], float], x:float) -> float:
    eps05 = math.sqrt(eps())
    if abs(x) < eps05:
        h = eps05
    else:
        h = eps05 * abs(x)

    return (fun(x+h) - fun(x-h))/( 2  *h)

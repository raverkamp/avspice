import math
import sys

def getargs():
    args = sys.argv
    if len(args) < 2:
        raise Exception("Expecting at least one argument")
    return (args[1], args[2:])

def drange(start, end, step=None):
    x = float(start)
    if step is None:
        s = 1.0
    else:
        s = step
    s = float(s)
    if s <=0:
        raise Exception("step <=0")
    while x < end:
        yield x
        x += s
    if x < end + s/2.0:
        yield end

def saw_tooth(freq,t):
    t = 1.0 * t * freq
    t = t % 1
    if t<0.5:
        return t*2
    else:
        return 2 - t*2


def explin(x: float, lcutoff: float, rcutoff:float):
    assert lcutoff  <= rcutoff, "cutoffs wrong"

    if x > rcutoff:
        return math.exp(rcutoff) +  (x-rcutoff) * math.exp(rcutoff)
    elif x < lcutoff:
        return math.exp(lcutoff) +  (x-lcutoff) * math.exp(lcutoff)
    else:
        return math.exp(x)

def dexplin(x:float, lcutoff:float, rcutoff:float):
    assert lcutoff  <= rcutoff, "cutoffs wrong"

    if x > rcutoff:
        return math.exp(rcutoff)
    elif x < lcutoff:
        return  math.exp(lcutoff)
    else:
        return math.exp(x)

def is_str_seq(l):
    for x  in l:
        if not isinstance(x, str):
            return False
    return True


def find_pos(v, x):
    """v is sorted double vector, x is double
       return largest i such that  x >= v[i], -1 if x< v[0]"""
    if x < v[0]:
        return -1
    i = len(v) -1
    while True:
        if i == -1:
            return -1
        if x >= v[i]:
            return i 
        i-=1
        
def linear_interpolate(x, y, t):
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


import math
import pprint as pp
import numpy as np
#import scipy.optimize

def reldiff(x,y):
    return abs(x-y) /  max(abs(x), abs(y))

def close_enough(v1,v2, abstol, reltol):
    for j in range(v1.size):
        x = v1[j]
        y = v2[j]
        if not (abs(x-y) < abstol  or reldiff(x,y) < reltol):
            return False
    return True


def solve(xstart, f, df, abstol, reltol, maxiter=20, x0 = None, alfa=None, verbose=False):
    iterations = 0
    if verbose:
        print("-----------------------------",xstart,alfa)
    x0 = xstart
    x = xstart
    if not alfa is None:
        fx0 = f(x0)
        dalfa = np.identity(len(x)) * alfa
    else:
        fx0 = None
        dalfa = None

    # has a root at x for alfa=1
    # alfa=0 equivalent to F(x)
    def fn(x):
        if alfa is None:
            return f(x)
        else:
            return f(x) + ((x-x0) - fx0) * alfa

    def dfn(x):
        if alfa is None:
            return df(x)
        else:
            return  df(x) + dalfa

    while True:
        if iterations > maxiter:
            return "Fail"
        iterations +=1
        y = fn(x)
        norm_y = np.linalg.norm(y)
        dfx = dfn(x)
        dx = np.linalg.solve(dfx, -y)
        if verbose:
            print("iteration {0}, norm_y={1}, norm_dx={2} ----".format(iterations, norm_y,  np.linalg.norm(dx)))

        a = 1
        k = 0
        while True:
            k = k + 1
            if k >= 20:
                return "fail"
            xn = x + a * dx
            yn = fn(xn)
            norm_y_n = np.linalg.norm(yn)
            if k >= 15 and verbose:
                print("#", k,norm_y_n, norm_y_n/norm_y, 1-a/2)
                print("++", xn)
            # if everything was linear we would expect norm_y / norm_y = 1-a
            if norm_y_n < abstol or norm_y_n/norm_y <= (1-a/4):
                break
            else:
                a = a/2
        if verbose:
            print(("norm_y_n", norm_y_n))
        if norm_y_n < abstol:
            return (xn, yn, dfx, iterations, norm_y_n)
        x = xn


def bisect(f, xl, xr):
    assert xl < xr, "xl < xr required!"
    fr = f(xr)
    fl = f(xl)
    print((fl, fr))
    assert f(xl) * f(xr) <0, "xl < xr required!"
    while True:
        xm = (xl + xr)/2
        if xm == xl or xm == xr:
            return xm
        fm = f(xm)
        print((xm, xl, xr))
        if (fm <=0 and fl <=0) or (fm >=0 and fl >=0):
            xl = xm
            fl = fm
        elif (fm <=0 and fr<=0) or (fm >=0 and fr >=0):
            xr = xm
            fr = fm
        else:
            raise Exception("Bug?")




def scipy_solve(xstart, f, df, abstol, reltol, maxiter=20, x0 = None, alfa=None):
    x0 = xstart
    x = xstart
    if not alfa is None:
        fx0 = f(x0)
        dalfa = np.identity(len(x)) * alfa
    else:
        fx0 = None
        dalfa = None

    # has a root at x for alfa=1
    # alfa=0 equivalent to F(x)
    def fn(x):
        if alfa is None:
            return f(x)
        else:
            return f(x) + ((x-x0) - fx0) * alfa

    def dfn(x):
        if alfa is None:
            return df(x)
        else:
            return  df(x) + dalfa

    res = scipy.optimize.fsolve(fn, x, fprime=dfn, full_output=True)
    print("@@@@@@@", type(res), res)
    (x, infodict, ier, mesg) = res

    if ier == 1:
        return (x, infodict["fvec"], infodict["nfev"],None,0)
    else:
        return mesg

import math
import pprint as pp
import numbers
import numpy as np


def reldiff(x,y):
    return abs(x-y) /  max(abs(x), abs(y))

def close_enough(v1,v2, abstol, reltol):
    for j in range(v1.size):
        x = v1[j]
        y = v2[j]
        if not (abs(x-y) < abstol  or reldiff(x,y) < reltol):
            return False
    return True

def solve_old(x0, f, df, abstol, reltol, maxiter=20):
    iterations = 0
    x = x0
    while True:
        if iterations > maxiter:
            break
        iterations +=1
        y = f(x)
        norm_y = np.linalg.norm(y)
        dfx = df(x)
        dx = np.linalg.solve(dfx, -y)
        print("iteration {0}, norm_y_n={1}, norm_dx={2} ----".format(iterations, norm_y,  np.linalg.norm(dx)))
        if iterations > -maxiter/2:
            k = 0
            while True:
                k = k+1
                if k > 20:
                    return "fail"
                xn = x + dx
                norm_y_n = np.linalg.norm(f(xn))
                if norm_y_n <= norm_y:
                    break
                print(("dn", k, norm_y,  norm_y_n))
                dx = dx/2
        else:
            xn = x + dx *alpha
        #print((xn, y, dfx))
        if (close_enough(x, xn, abstol, reltol)
            or norm_y < abstol
            or iterations > maxiter -1):
            return (xn, y, dfx, iterations, norm_y)
        x = xn
    return "Fail"


def solve(x0, f, df, abstol, reltol, maxiter=20):
    iterations = 0
    x = x0
    while True:
        if iterations > maxiter:
            return "Fail"
        iterations +=1
        y = f(x)
        norm_y = np.linalg.norm(y)
        dfx = df(x)
        dx = np.linalg.solve(dfx, -y)
        print("iteration {0}, norm_y={1}, norm_dx={2} ----".format(iterations, norm_y,  np.linalg.norm(dx)))
        a = 1
        k = 0
        while True:
            k = k + 1
            if k >= 10:
                return "fail"
            xn = x + a * dx
            norm_y_n = np.linalg.norm(f(xn))
            if k >= -2:
                print("#", k,norm_y_n, norm_y_n/norm_y, 1-a/4)
            # if everything was linear we would expect norm_y / norm_y = 1-a
            if norm_y_n/norm_y <= (1-a/2):
                break
            else:
                a = a/2
        
        if (close_enough(x, xn, abstol, reltol)
            or norm_y < abstol):
            return (xn, y, dfx, iterations, norm_y)
        x = xn


def solvea(xstart, f, df, abstol, reltol, maxiter=20, x0 = None, alfa=None):
    iterations = 0
    print("-----------------------------")
    print(xstart)
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
            if k >= -2:
                print("#", k,norm_y_n, norm_y_n/norm_y, 1-a/2)
            # if everything was linear we would expect norm_y / norm_y = 1-a
            if norm_y_n < abstol or norm_y_n/norm_y <= (1-a/4):
                break
            else:
                a = a/2
        print(("norm_y_n", norm_y_n))
        if norm_y_n < abstol:
            return (xn, yn, dfx, iterations, norm_y_n)
        x = xn

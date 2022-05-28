import collections
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

BasicSolution = collections.namedtuple('Solution', ['x', 'y', 'dfx', 'iterations', 'norm_y'])

def solve(xstart, f, df, abstol, reltol, maxiter=20, alfa=None, verbose=False):
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

    # has a root at x0 for alfa=1
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

    y = fn(x)
    norm_y = np.linalg.norm(y)

    while True:
        if iterations > maxiter:
            return f"Fail #iterations={iterations}"
        iterations +=1

        dfx = dfn(x)
        dx = np.linalg.solve(dfx, -y)
        if verbose:
            print(f"iteration {iterations}, norm_y={norm_y}, norm_dx={np.linalg.norm(dx)} ----")
        xn = x + dx
        yn =  fn(xn)
        norm_y_n = np.linalg.norm(yn)
        #if iterations > 100:
        #    print("iteration", norm_y, x-xn)
        if close_enough(x, xn, abstol, reltol):
            return BasicSolution(x=xn, y=yn, dfx=dfx, iterations=iterations, norm_y=norm_y_n)
        a = 1
        k = 0

        while True:
            # is there an improvement in the residual error?
            if norm_y_n < abstol or norm_y_n/norm_y <= (1-a/4):
                x = xn
                break
            k = k + 1
            if k >= 20:
                x =  xn
                return "Fail no improvment"
            a = a/2
            xn = x + a * dx
            yn = fn(xn)
            norm_y_n = np.linalg.norm(yn)
        x = xn
        y = yn
        norm_y = norm_y_n

Solution = collections.namedtuple('Solution', ['solution', 'y', 'dfx', 'iterations', 'norm_y'])

def solve_alfa(xstart, f, df, abstol, reltol, maxiter=20, verbose=False):
    solution_vec = xstart
    res = solve(solution_vec, f, df, abstol, reltol, maxiter)
    if not isinstance(res, str):
        return res
    alfa = 0.5

    for _ in range(20):
        res = solve(solution_vec, f, df, abstol, reltol, maxiter, alfa=alfa)
        if not isinstance(res, str):
            print(res)
            solution_vec = res.x
            break
        alfa = (alfa + 1) / 2
    if isinstance(res,str):
        print("failed getting initial solution")
        return "failed getting initial solution"

    while True:
        alfa = max(alfa / 1.1, 0)
        res = solve(solution_vec, f, df, abstol, reltol, maxiter,
                            alfa=alfa)
        if isinstance(res, str):
            print(f"alfa={alfa}")
            return res
        if alfa <=0:
            break
        solution_vec = res[0]
    return res
    #    (sol, y, dfx, iterations, norm_y) = res

def bisect(f, xl, xr):
    assert xl < xr, "xl < xr required!"
    fr = f(xr)
    fl = f(xl)
    assert f(xl) * f(xr) <0, "xl < xr required!"
    while True:
        xm = (xl + xr)/2
        if xm == xl or xm == xr:
            return xm
        fm = f(xm)
        if (fm <=0 and fl <=0) or (fm >=0 and fl >=0):
            xl = xm
            fl = fm
        elif (fm <=0 and fr<=0) or (fm >=0 and fr >=0):
            xr = xm
            fr = fm
        else:
            raise Exception("Bug?")

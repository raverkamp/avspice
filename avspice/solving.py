"""the solver module of avspice"""

from typing import Optional, Callable, Union, NamedTuple

import numpy as np
import numpy.typing as npt



def reldiff(x:float,y:float) ->float:
    return abs(x-y) /  max(abs(x), abs(y))

def close_enough(v1: npt.NDArray[np.float64],
                 v2: npt.NDArray[np.float64],
                 abstol: float,
                 reltol:float) -> bool:
    for j in range(v1.size):
        x = v1[j]
        y = v2[j]
        if not (abs(x-y) < abstol  or reldiff(x,y) < reltol):
            return False
    return True

BasicSolution = NamedTuple('BasicSolution',
                           [('x', npt.NDArray[np.float64]),
                            ('y',npt.NDArray[np.float64]),
                            ('dfx', npt.NDArray[np.float64]),
                            ('iterations', int),
                            ('norm_y', float)])

def solve(xstart: npt.NDArray[np.float64],
          f: Callable[[npt.NDArray[np.float64]],npt.NDArray[np.float64]],
          df: Callable[[npt.NDArray[np.float64]],npt.NDArray[np.float64]],
          abstol:float,
          reltol:float,
          maxiter:int=20,
          alfa:Optional[float]=None,
          verbose:bool=False) ->Union[BasicSolution,str]:
    iterations = 0
    if verbose:
        print("-----------------------------",xstart,alfa)
    x0 = xstart
    x = xstart
    if not alfa is None:
        fx0 = f(x0)
        dalfa = np.identity(len(x)) * alfa
    else:
        fx0 = np.zeros(len(x))
        dalfa = np.identity(len(x))

    # has a root at x0 for alfa=1
    # alfa=0 equivalent to F(x)
    def fn(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        if alfa is None:
            return f(x)
        else:
            return f(x) + ((x-x0) - fx0) * alfa

    def dfn(x:npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        if alfa is None:
            return df(x)
        else:
            return  df(x) + dalfa

    y = fn(x)
    norm_y = float(np.linalg.norm(y))

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
        norm_y_n = float(np.linalg.norm(yn))
        #if iterations > 100:
        #    print("iteration", norm_y, x-xn)
        if close_enough(x, xn, abstol, reltol):
            return BasicSolution(x=xn, y=yn, dfx=dfx, iterations=iterations, norm_y=norm_y_n)

        a:float = 1
        k:int = 0

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
            norm_y_n = float(np.linalg.norm(yn))
        x = xn
        y = yn
        norm_y = norm_y_n

def solve_alfa(xstart:npt.NDArray[np.float64],
               f: Callable[[npt.NDArray[np.float64]],npt.NDArray[np.float64]],
               df: Callable[[npt.NDArray[np.float64]],npt.NDArray[np.float64]],
               abstol:float,
               reltol:float,
               maxiter:int=20,
               verbose:bool=False) -> Union[BasicSolution,str]:
    solution_vec = xstart
    res = solve(solution_vec, f, df, abstol, reltol, maxiter)
    if not isinstance(res, str):
        return res
    alfa = 0.5

    for _ in range(20):
        res = solve(solution_vec, f, df, abstol, reltol, maxiter, alfa=alfa)
        if not isinstance(res, str):
            solution_vec = res.x
            if verbose:
                print(f"got start with alfa={alfa}")
            break
        alfa = (alfa + 1) / 2
    if isinstance(res,str):
        if verbose:
            print("failed getting initial solution")
        return "failed getting initial solution"

    while True:
        alfa = max(alfa / 1.1, 0)
        if alfa < 1e-3:
            alfa = 0
        res = solve(solution_vec, f, df, abstol, reltol, maxiter,
                            alfa=alfa)
        if isinstance(res, str):
            if verbose:
                print(f"fail at alfa={alfa}")
            return res
        if alfa <=0:
            break
        solution_vec = res[0]
    return res
    #    (sol, y, dfx, iterations, norm_y) = res

def bisect(f: Callable[[float],float],
           xl:float,
           xr:float)->float:
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

import matplotlib as mp
import matplotlib.pyplot as plt
from avspice import *
import argparse
import sys
from avspice.util import *

from example_circuits import create_blinker


def blinker_static(args):
    net = create_blinker()
    ana = Analysis(net)

    base_vca = -0.4   #0.397 #-5 # -8.25


    sol = None
    res = ana.analyze(maxit=50, start_solution_vec=sol, capa_voltages={"ca": base_vca},
                      variables={"vc": 9},
                      start_voltages= {
                          "t1.C": 4,
                          "t2.C": 0.1,
                          "t1.B": 0.1})
    res.display()


def try_random_start(args):
    import random
    net = create_blinker()
    ana = Analysis(net)
    base_vca = args.vca
    res = ana.analyze(maxit=50, start_solution_vec=None, capa_voltages={"ca": base_vca},
                      variables={"vc": 9},
                      start_voltages= {
                          "t1.C": args.t1c,
                          "t1.B": args.t1b
                      })
    if isinstance(res, str):
        print("fail")
    else:
        for s in ["t1.C", "t1.B", "t2.C", "t2.B"]:
            print("{0} V = {1}   I={2}".format(s, res.get_voltage(s), res.get_current(s)))


def blinker3(args):
    net = create_blinker(transistor_gain=args.gain, cutoff=args.cutoff)
    ana = Analysis(net)

    base_vca = args.vca

    variables = {"capa": args.capa, "vc": 10}

    l=[]
    for (k,v) in sorted(variables.items(), key=lambda kv: kv[1]):
        l.append(f"{k}={v}")
    l.append(f"gain={args.gain}")
    fig_title = ", ".join(l)


    res = ana.transient(args.end, 0.001, capa_voltages={"ca": base_vca},
                        variables=variables,
                        maxit=200,
                        start_voltages= {
                            "t1.C": args.t1c,
                            "t1.B": args.t1b,
                            "t2.C": args.t2c
                        })

    (time,volts,currs) = pivot(res)
    fig, ((a1,a2, a3), (a4, a5, a6), (a7,a8,a9)) = plt.subplots(3,3)
    a1.plot(time,currs["t1.C"],label="curr t1.C")
    a1.legend()
    a2.plot(time,currs["t2.C"], label="curr t2.C")
    a2.legend()
    a3.plot(time,currs["ca.p"], label="curr ca.p")
    a3.legend()

    a4.plot(time,volts["t1.C"],label="V(t1.C)")
    a4.plot(time,volts["t2.C"],label="V(t2.C")
    a4.plot(time,volts["t1.B"],label="V(t1.B)")
    a4.legend()
    a5.plot(time,currs["t1.B"],label="curr t1.B")
    a5.legend()
    a6.plot(time,currs["t1.E"],label="curr t1.E")
    a6.legend()
    a7.plot(time, volts["ca.p"] - volts["ca.n"], label="voltage capa")
    a7.legend()
    a8.plot(time,currs["t2.B"],label="curr t2.B")
    a8.legend()

    a9.plot(time,currs["d1.p"],label="curr d1.B")
    a9.legend()

    fig.suptitle(fig_title)

    plt.show()



def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(required=True)

    parser_s = subparsers.add_parser('s')
    parser_s.set_defaults(func=blinker_static)

    parser_sol = subparsers.add_parser('sol')
    parser_sol.set_defaults(func=try_random_start)
    parser_sol.add_argument('vca', type=float)
    parser_sol.add_argument('-t1c', type=float, default=0)
    parser_sol.add_argument('-t1b', type=float, default=0)
   
    parser_b3 = subparsers.add_parser('b3')
    parser_b3.set_defaults(func=blinker3)
    parser_b3.add_argument('-t1c', type=float, default=0)
    parser_b3.add_argument('-t1b', type=float, default=0)
    parser_b3.add_argument('-t2c', type=float, default=0)
    parser_b3.add_argument('-vca', type=float, default=0)
    parser_b3.add_argument('-capa', type=float, default=10e-6)
    parser_b3.add_argument('-gain', type=float, default=100)
    parser_b3.add_argument('-end', type=float, default=3.6)
    parser_b3.add_argument('-cutoff', type=float, default=40)

    args = parser.parse_args()
    args.func(args)

sys.exit(main())

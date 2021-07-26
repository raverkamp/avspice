import matplotlib as mp
import matplotlib.pyplot as plt
from spice import *
import argparse
import sys
from util import *

import math


def kennlinie(args):
    print(("kennlinie", args))
    tt = NPNTransistor(None, "", 1e-12, 25e-3, 100, 10)
    net = Network()
    t1 = net.addComp("T1", tt)
    vc = net.addV("vc", Variable("vc"))
    cb = net.addC("cb", Variable("cb"))
    rb = net.addR("rb", 10e3)
    connect(t1.C, vc.p)
    connect(t1.E, net.ground)
    connect(vc.n, net.ground)
    connect(cb.p, rb.p)
    connect(rb.n, t1.B)
    connect(cb.n, net.ground)
    
    ana = Analysis(net)
    ce_voltages = list(drange(0, 10, 0.01))
    (fig, ax) = plt.subplots(1)
    for i in [ 50e-6, 100e-6, 200e-6, 400e-6]:
        y = []
        x = []
        sol = None
        for v in ce_voltages:
            res = ana.analyze(maxit=30, start_solution_vec=sol, variables={"vc": v, "cb": i})
            if isinstance(res, str):
                break
            x.append(v)
            y.append(res.get_current(t1.C))
            sol = res.solution_vec
        ax.plot(x,y,label=("IC for IB={0}".format(i)))
    ax.legend()
    fig.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(prog='Datenblatt')
    subparsers = parser.add_subparsers(help='sub-command help', dest='subparser_name')

    parser_k = subparsers.add_parser('k', help='k help')
    parser_k.set_defaults(func=kennlinie)
    
    args = parser.parse_args()
    args.func(args)
    
#     parser_a.add_argument('bar', type=int, help='bar help')
#     parser_b = subparsers.add_parser('b', help='b help')
#     parser_b.add_argument('--baz', choices='XYZ', help='baz help')

main()
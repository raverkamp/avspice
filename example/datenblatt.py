import matplotlib as mp
import matplotlib.pyplot as plt
from avspice import *
import argparse
import sys
from avspice.util import *
from avspice import ncomponents
import math


def kennlinie(args):
    tt = NPNTransistor("", 1e-12, 25e-3, 100, 10)
    net = Circuit()
    t1 = net.add_component("t1", tt, ("B", "v", "0")) 
    net.addV("vc", Variable("vc"), "v", "0")
    net.addV("cb", 1, "vb", "0")
    net.addR("rb", Variable("rb"), "vb", "B")
    
    ana = Analysis(net)
    ce_voltages = list(drange(0, 2, 0.01))
    (fig, ax) = plt.subplots(1)
    for i in [ 50e-6, 100e-6, 200e-6, 400e-6]:
        y = []
        x = []
        sol = None
        for v in ce_voltages:
            res = ana.analyze(maxit=30, start_solution_vec=sol, variables={"vc": v, "rb": 1/i})
            if isinstance(res, str):
                break
            x.append(v)
            y.append(res.get_current("t1.C"))
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
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_usage()

main()

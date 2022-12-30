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
    ax.set_ylabel("Current(C)")
    ax.set_xlabel("Voltage(CE)")
    fig.suptitle("NPN Curve")
    fig.tight_layout()
    plt.show()

def zkennlinie(args):
    zd = ncomponents.NZDiode(5, 1e-8, 25e-3, 1e-8, 25e-3)
    (fig, (ax1, ax2, ax3)) = plt.subplots(3)
    fig.suptitle('Curves of z-diode', fontsize=16)
    x = []
    y = []
    dy = []
    sy = []
    for v in drange(-5.5,0.5,0.001):
        cu = zd.current(v)
        dcu = zd.diff_current(v)
        x.append(v)
        y.append(cu)
        dy.append(dcu)
        sy.append(math.copysign(1,cu))
    ax1.plot(x,y)
    ax1.set_title("current")
    ax2.plot(x,dy)
    ax2.set_title("d current")
    ax3.plot(x, sy)
    ax3.set_title("sign(current)")
    plt.show()

def fkennlinie(args):
    x = []
    y = []
    fet = ncomponents.NFET(1)
    (fig, ax1) = plt.subplots(1)
    for vgs in [1.01, 2, 3, 4, 5]:
        x = []
        y = []
        for vds in drange(0,7,0.01):
            x.append(vds)
            y.append(fet.IS(vgs,vds))
        ax1.plot(x,y,label=f"v(gs)={vgs}")

    ax1.set_xlabel("voltage(ds)")
    ax1.set_ylabel("current(ds)")
    ax1.legend()
    plt.show()
       

def main():
    parser = argparse.ArgumentParser(prog='Datenblatt')
    subparsers = parser.add_subparsers(help='sub-command help', dest='subparser_name')

    parser_k = subparsers.add_parser('npn', help='npn curce')
    parser_k.set_defaults(func=kennlinie)

    parser_z = subparsers.add_parser('zdiode', help='z diode curve')
    parser_z.set_defaults(func=zkennlinie)


    parser_fet = subparsers.add_parser('fet', help='fet curve')
    parser_fet.set_defaults(func=fkennlinie)

    
    args = parser.parse_args()
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_usage()

main()

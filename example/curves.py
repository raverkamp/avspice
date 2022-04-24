import matplotlib as mp
import matplotlib.pyplot as plt
from avspice import *
import argparse
import sys
from avspice.util import *
from avspice  import ncomponents

import math

import argparse


nnpntransistor = ncomponents.NNPNTransistor(1e-12, 25e-3, 100, 10,-40, 40)

npntransistor = NPNTransistor("", 1e-12, 25e-3, 100, 10)


def plot1(args):

    t = nnpntransistor
    fig, (ax1, ax2) = plt.subplots(2)
    vbs = list(drange(0.2, 0.5, 0.01))
    ve = 0
    vc = 0
    ie = [t.IE(vb-ve, vb-vc) for vb in vbs]
    ib = [t.IB(vb-ve, vb-vc) for vb in vbs]
    ic = [t.IC(vb-ve, vb-vc) for vb in vbs]

    ax1.set_title("vc = ve")
    ax1.set_ylabel("current")
    ax1.set_xlabel("vb")
    ax1.plot(vbs,ie, label="ie")
    ax1.plot(vbs,ib, label="ib")
    ax1.plot(vbs,ic, label="ic")
    ax1.legend()

    ie = []
    ib = []
    ic = []
    for vb in vbs:
        ve = 0
        vc = vb
        ie.append(t.IE(vb-ve, vb-vc))
        ib.append(t.IB(vb-ve, vb-vc))
        ic.append(t.IC(vb-ve, vb-vc))
    ax2.set_title("vc = vb")
    ax2.set_ylabel("current")
    ax2.set_xlabel("vb")
    ax2.plot(vbs,ie, label="ie")
    ax2.plot(vbs,ib, label="ib")
    ax2.plot(vbs,ic, label="ic")
    ax2.legend()

    plt.ion()
    plt.show()
    input()

def plot2(args):
    x = list(drange(0, 2, 0.01))
    y = []
    z = []
    sol = None
    iy = []

    tt = npntransistor
 
    net = Circuit()
    net.addV("vc", 5, "v", "0")
    net.addV("vb", Variable("vb"), "vb","0")
    net.addR("rc", 100, "v", "C")
    net.addR("rb", 1e3, "vb", "B")
    net.add_component("t1", tt, ("B", "C", "0"))
    ana = Analysis(net)
    
    for vb in x:
        res = ana.analyze(maxit=30, start_solution_vec=sol, variables={"vb": vb})
        if isinstance(res, str):
            print("no covergence at: {0}".format(v))
            y.append(None)
            z.append(None)
            iy.appned(None)
            sol = None
        else:
            y.append(res.get_current("rc.p"))
            z.append(res.get_current("rb.p"))
            iy.append(res.iterations)
            sol = res.solution_vec
    fig, (ax1, ax2) = plt.subplots(2)
    ax1.plot(x,y, color="black", label="I(E)")
    ax1.set_xlabel("V(B,E)")
    ax1.set_ylabel("current")
    ax1.legend()
    
    ax2.set_xlabel("V(B,E)")
    ax2.set_ylabel("current")    
    ax2.plot(x,z, color="black", label="I(B)")
    ax2.legend()
    
    fig.tight_layout()
    plt.show()
    
def main():
    (cmd, args) = getargs()
    if cmd == "1":
        plot1(args)
    elif cmd == "2":
        plot2(args)
    else:
        raise Exception("unknown commnd: {0}".format(cmd))

main()


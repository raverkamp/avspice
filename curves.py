import matplotlib as mp
import matplotlib.pyplot as plt
from spice import *
import argparse
import sys
from util import *

import math

import argparse


npntransistor = NPNTransistor(None, "", 1e-12, 25e-3, 100, 10)



def plot1(args):

    t = npntransistor
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
 
    net = Network()
    v = net.addV("vc", 5)
    vb = net.addV("vb", Variable("vb"))
    rc = net.addR("rc", 100)
    rb = net.addR("rb", 1e3)
    t1 = net.addComp("T1", tt)
    connect(v.p, rc.p)
    connect(rc.n, t1.C)
    connect(v.n, net.ground)
    connect(vb.p, rb.p)
    connect(rb.n, t1.B)
    connect(vb.n, net.ground)
    connect(t1.E, net.ground)
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
            y.append(res.get_current(rc.p))
            z.append(res.get_current(rb.p))
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


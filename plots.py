import matplotlib as mp
import matplotlib.pyplot as plt
from spice import *
import argparse
import sys
from util import *

import math

from ncomponents import NNPNTransistor



def plot1(args):
    # das passt alles nicht
    # oder doch: Basis Spannug sollte über der vom Kollektor liegen
    t = NNPNTransistor(1e-12, 25e-3, 100, 10, -40, 40)
    vbl = [0, 0.1, 0.2, 0.3,0.4]
    k = math.ceil(math.sqrt(len(vbl)))
    (f,plts) = plt.subplots(k,k)
    ve = 0
    x = list(drange(0, 0.3, 0.01))
    i = 0
    for vb in vbl:
        ax = plts[i//k][i%k]
        ie = [t.IE(vb-ve, vb-vc) for vc in x]
        ib = [t.IB(vb-ve, vb-vc) for vc in x]
        te = "npn transistor, vb={0}, ve={1}".format(vb, ve)
        ax.set_title(te)
        ax.set_ylabel("current")
        ax.set_xlabel("vc")
        ax.plot(x,ie, label="ie")
        ax.plot(x,ib, label="ib")
        ax.legend()
        i+=1

    plt.ion()
    plt.show()
    input()

def plot2(args):
    t = NNPNTransistor(1e-12, 25e-3, 100, 10, -40, 40)

    vc = 2
    ve = 0
    x = list(drange(-0.5,3.5, 0.01)    )
    ie = [t.IE(vb-ve, vb-vc) for vb in x]
    ib = [t.IB(vb-ve, vb-vc) for vb in x]
    _, _ = plt.subplots()  # Create a figure containing a single axes.
    plt.plot(x,ie, color="black")
    plt.plot(x,ib, color="green")
    plt.show()
    input()

def plot3(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('cpot', type=float)
    parser.add_argument('cutoff', type=float)

    args = parser.parse_args(args)
    t = NNPNTransistor(1e-12, 25e-3, 100, 10, -40, 40)
    x = list(drange(-0.5, 3.5, 0.01))
    vc = args.cpot
    ve = 0
    ie = [t.IE(vb-ve, vb-vc) for vb in x]
    ib = [t.IB(vb-ve, vb-vc) for vb in x]
    ic = [t.IC(vb-ve, vb-vc) for vb in x]
    plt.plot(x,ie, color="black")
    plt.plot(x,ib, color="green")
    plt.plot(x,ic, color="blue")
    plt.show()
    input()


def plot4(args):
    x = list(drange(-2, 10, 0.01))
    y = []
    z = []
    sol = None
    iy = []

    tt = NPNTransistor(None, "", 1e-12, 25e-3, 100, 10)

    net = Network()
    vc = net.addV("vc", 2)
    vb = net.addV("vb", Variable("vb"))
    re = net.addR("re", 100)
    rb = net.addR("rb", 10e3)
    t1 = net.addComp("T1", tt)
    connect(vc.p, t1.C)
    connect(vc.n, net.ground)
    connect(vb.p, rb.p)
    connect(rb.n, t1.B)
    connect(vb.n, net.ground)
    connect(t1.E, re.p)
    connect(re.n, net.ground)
    ana = Analysis(net)
    for v in x:
        res = ana.analyze(maxit=30, start_solution_vec=sol, variables={"vb": v})
        if isinstance(res, str):
            print("no covergence at: {0}".format(v))
            y.append(None)
            z.append(None)
            iy.appned(None)
            sol = None
        else:
            y.append(res.get_current(t1.E))
            z.append(res.get_current(t1.B))
            iy.append(res.iterations)
            sol = res.solution_vec
    fig, (ax1,ax) = plt.subplots(2)
    ax1.plot(x,y, color="black", label="I(E)")
    ax2 = ax1.twinx()
    ax2.plot(x,z, color="green", label="I(B)")
    ax1.legend()
    ax2.legend()

    fig.tight_layout()
    ax.plot(x, iy)
    ax.set_title("Iterations")
    plt.show()
    #input()

def plot5(args):
    x = list(drange(-2, 10, 0.01))
    y = []
    z = []
    sol = None
    iy = []

    tt = NPNTransistor(None, "", 1e-12, 25e-3, 100, 10)
    net = Network()
    vc = net.addV("vc", 5)
    rc = net.addR("rc", 1e2)
    rb = net.addR("rb", Variable("rb"))
    t1 = net.addComp("T1", tt)
    connect(vc.p, rc.p)
    connect(rc.n, t1.C)
    connect(vc.p, rb.p)
    connect(rb.n, t1.B)
    connect(vc.n, net.ground)
    connect(t1.E, net.ground)

    ana = Analysis(net)
    sol = None
    lrb = []
    lvb = []
    for vrb in drange(1e3,1e6,1000):
        res = ana.analyze(maxit=30, start_solution_vec=sol, variables={"rb": vrb})
        if isinstance(res, str):
            print("no covergence at: {0}".format(vrb))
            y.append(None)
            z.append(None)
            iy.append(None)
            sol = None
        else:
            lrb.append(vrb)
            lvb.append(res.get_voltage (t1.B))
            sol = res.solution_vec
    fig, (ax1,ax2) = plt.subplots(2)
    ax1.plot(lrb,lvb,color="black")
    fig.tight_layout()
    plt.show()
    #input()


def emitter(args):
    tt = NPNTransistor(None, "", 1e-12, 25e-3, 100, 10)
    nw = Network()
    net = Network()
    vc = net.addV("vc", 5)
    r1 = net.addR("r1", 2000)
    r2 = net.addR("r2", 2000)


    vb = net.addV("vb", Variable("vb"))
    rc = net.addR("rc", 1e3)
    rb = net.addR("rb", 1e5)

    t1 = net.addComp("T1", tt)

    connect(vc.p, rc.p)
    connect(vc.n, net.ground)
    connect(rc.n, t1.C)
    connect(t1.E, net.ground)
    connect(r1.p, vc.p)
    connect(r1.n, r2.p)
    connect(r2.n, net.ground)
    connect(r1.n, t1.B)

    connect(vb.p, rb.p)
    connect(vb.n, net.ground)
    connect(rb.n, t1.B)


    y = []
    z = []
    ana = Analysis(net)
    sol = None
    x = list(drange(0,0.1,0.001))
    for v in x:
        res = ana.analyze(maxit=30, start_solution_vec=sol, variables={"vb": v})
        if isinstance(res, str):
            print("no covergence at: {0}".format(v))
            y.append(None)
            sol = None
        else:
            y.append(res.get_voltage(t1.C))
            z.append(res.get_voltage(t1.B))
            sol = res.solution_vec
    fig, (a1,a2) = plt.subplots(2)
    a1.plot(x,y, color="blue")
    a2.plot(x,z, color="blue")
    print(z)
    plt.show()


def saw1(args):
    import util
    net = Network()
    r = 100000
    capa = 10e-6
    net.addSawV("vc", 1, 1, "v", "0")
    net.addR("r1", r, "v","ca")
    net.addCapa("ca", capa, "ca", "0")

    ana = Analysis(net)
    res = ana.transient(9,0.0001, capa_voltages={"ca":0})

    time = []
    ca_p = []
    for (t,v,c) in res:
        time.append(t)
        ca_p.append(v["ca.p"])

    (f, p1) = plt.subplots(1)
    p1.set_title("Voltage at capacitor, R={0}, CAPA={1}".format(r,capa))
    p1.plot(time, ca_p)
    plt.show()


def emitterschaltung(args):
    #   https://www.elektronik-kompendium.de/sites/slt/0204302.htm
    import util
    tt = NPNTransistor("", 1e-12, 25e-3, 100, 10)
    net = Network()
    net.addV("vc", 10, "vcc", "0")
    net.addSawV("ve", 0.1, 10, "ve", "0")

    net.add_component("tr", tt, ("B", "C","0"))

    net.addCapa("ce", 100e-6,"ve","B")
    #ce = net.addR("ce", 1)
    net.addCapa("ca", 1e-6, "C", "last")

    net.addR("r1", 10e3, "vcc", "B")
    net.addR("r2", 0.5e3, "B", "0")

    net.addR("rc",1000, "vcc", "C")

    rl = net.addR("rl", 100, "last", "0")

    ana = Analysis(net)
    res = ana.transient(0.4,0.0001, capa_voltages={"ca":0, "ce":0})
    time = []
    va = []
    ca = []
    ctb = []
    vtb = []
    for (t,v,c) in res:
        time.append(t)
        va.append(v["rl.p"])
        ca.append(c["rl.p"])
        ctb.append(c["tr.B"])
        vtb.append(v["tr.B"])
    (fig, (p1, p2, p3, p4)) = plt.subplots(4)
    p1.plot(time, ctb)
    p1.set_title("current base")
    p2.plot(time, vtb)
    p2.set_title("voltage base")

    p3.plot(time, va)
    p3.set_title("voltage out")
    p4.plot(time, ca)
    p4.set_title("current out")
    fig.tight_layout()

    plt.show()

def rlc(args):
    net = Network()
    i0 = 0.1
    rv = 100
    indu = 1
    capa = 10e-6

    net.addR("r1", rv, "1", "2")
    net.addCapa("ca", capa, "2", "0")
    net.addInduc("ind", indu, "0", "1")

    ana = Analysis(net)
    res = ana.transient(0.1,0.0001, induc_currents={"ind": 1}, capa_voltages={"ca":0} )

    ip = []
    iv = []
    time = []
    for (t,v,c) in res:
        time.append(t)
        ip.append(c["r1.p"])
        iv.append(v["ind.p"] - v["ind.n"])

    (fig, ax) = plt.subplots(1)
    ax.plot(time,ip, label="current", color="blue")
    f = 1/(math.sqrt(capa*indu) * 2 * math.pi)
    ax.set_title(f"RLC r={rv}, c={capa}, l={indu}, freq={f}")
    import  matplotlib.ticker
    formatterx = matplotlib.ticker.EngFormatter("s")
    formattery = matplotlib.ticker.EngFormatter("A")
    ax.xaxis.set_major_formatter(formatterx)
    ax.xaxis.set_minor_formatter(formatterx)
    ax.yaxis.set_major_formatter(formattery)
    ax.yaxis.set_minor_formatter(formattery)


    ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.plot(time,iv,label="voltage across inductor", color="red")
    formattery2 = matplotlib.ticker.EngFormatter("V")
    ax2.yaxis.set_major_formatter(formattery2)
    ax2.yaxis.set_minor_formatter(formattery2)
    fig.legend()
    plt.show()

def tricky(args):
    # loading a capacitor and an switching a transistor base
    # when loaded
    capa = 100e-6
    net = Network()
    tt = NPNTransistor("", 1e-12, 25e-3, 100, 10)
    net.add_component("t", tt, ("B", "C", "0"))
    net.addV("v", 9, "v", "0")
    net.addR("r1", 100, "v", "C")
    net.addR("r2", 50e3, "C", "B")
    net.addCapa("ca", capa, "B", "0")
        
    ana = Analysis(net)
    res = ana.transient(0.6,0.0001, capa_voltages={"ca":0.0} )
    (time,volts,currs) = pivot(res)
    (fig, (a1, a2, a3)) = plt.subplots(3)
    a1.set_title("curr(t.E)")
    a1.plot(time, currs["t.E"])
    a2.set_title("curr(ca)")
    a2.plot(time, currs["ca.p"])
    a3.set_title("volts(B)")
    a3.plot(time, volts["t.B"])
    fig.legend()
    plt.show()

def main():
    (cmd, args) = getargs()
    if cmd == "1":
        plot1(args)
    elif cmd == "2":
        plot2(args)
    elif cmd == "3":
        plot3(args)
    elif cmd == "4":
        plot4(args)
    elif cmd == "5":
        plot5(args)
    elif cmd == "e":
        emitter(args)
    elif cmd == "saw":
        saw1(args)
    elif cmd == "emitter":
        emitterschaltung(args)
    elif cmd == "rlc":
        rlc(args)
    elif cmd == "tricky":
        tricky(args)
    else:
        raise Exception("unknown commnd: {0}".format(cmd))

main()

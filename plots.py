import matplotlib as mp
import matplotlib.pyplot as plt
from spice import *
import argparse
import sys
from util import *

import math




def plot1(args):
    # das passt alles nicht
    # oder doch: Basis Spannug sollte über der vom Kollektor liegen
    t = NPNTransistor(None, "", 1e-12, 25e-3, 100, 10)
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
    t = NPNTransistor(None, "", 1e-12, 25e-3, 100, 10)

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
    t = NPNTransistor(None, "", 1e-12, 25e-3, 100, 10)
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

def capa(args):
    net = Network()
    vc = net.addV("vc", 2)
    r = net.addR("r", 1e4)
    c = net.addCapa("ca", 100e-6)
    connect(vc.p, c.p)
    connect(c.n, r.p)
    connect(r.n, vc.n)
    connect(vc.n, net.ground)
    ana = Analysis(net)
    ch = 0

    xs = []
    ys = []
    vcp = []
    vcn = []
    s = 0.2
    x = 0
    while x < 10:
        res = ana.analyze(maxit=30, charges={"ca": ch })
        ica = res.get_current(c.p)
        ch += s * ica
        x += s
        xs.append(x)
        ys.append(ch)
        vcp.append(res.get_voltage("ca.p"))
        vcn.append(res.get_voltage("ca.n"))
    fig, (a1,a2) = plt.subplots(2)
    a1.set_title("charge")
    a1.plot(xs,ys, color="blue")
    a2.set_title("voltage1 ,vcp blue, vcn red")
    a2.plot(xs, vcp, color="blue")
    a2.plot(xs, vcn, color="red")
    plt.show()

def saw1(args):
    import util
    net = Network()
    r = 100000
    capa = 10e-6
    vc = net.addSawV("vc", 1, 1)
    r1 = net.addR("r1", r)
    ca = net.addCapa("ca", capa)
    connect(vc.p, r1.p)
    connect(r1.n, ca.p)
    connect(ca.n, vc.n)
    connect(vc.n, net.ground)

    ana = Analysis(net)
    res = ana.transient(9,0.0001, capa_voltages={"ca":0})
    return
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
    tt = NPNTransistor(None, "", 1e-12, 25e-3, 100, 10)
    net = Network()
    vc = net.addV("vc", 10)

    ve = net.addSawV("ve", 0.1, 10)

    ce = net.addCapa("ce", 100e-6)
    #ce = net.addR("ce", 1)
    ca = net.addCapa("ca", 10e-6)

    r1 = net.addR("r1", 10e3)
    r2 = net.addR("r2", 0.5e3)

    rc = net.addR("rc",1000)

    tr = net.addComp("tr", tt)

    rl = net.addR("rl", 100)

    connect(vc.p, r1.p)
    connect(vc.n, net.ground)
    connect(r1.n, r2.p)
    connect(r2.n, net.ground)

    connect(r1.n, tr.B)

    connect(ve.p, ce.p)
    connect(ve.n, net.ground)
    connect(ce.n, tr.B)

    connect(rc.p, vc.p)
    connect(rc.n, tr.C)

    connect(tr.E, net.ground)

    connect(tr.C, ca.p)

    connect(ca.n, rl.p)
    connect(rl.n, net.ground)

    ana = Analysis(net)
    res = ana.transient(0.4,0.00001, capa_voltages={"ca":0, "ce":0})
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

    r = net.addR("r1", rv)
    ca = net.addCapa("ca", capa)
    ind = net.addInduc("ind", indu)

    connect(r.n, ca.p)
    connect(ca.n, ind.p)
    connect(ind.n, r.p)
    connect(ind.n,net.ground)

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
    elif cmd == "c":
        capa(args)
    elif cmd == "saw":
        saw1(args)
    elif cmd == "emitter":
        emitterschaltung(args)
    elif cmd == "rlc":
        rlc(args)
    else:
        raise Exception("unknown commnd: {0}".format(cmd))

main()

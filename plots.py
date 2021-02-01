import matplotlib as mp
import matplotlib.pyplot as plt
from spice import *
import argparse
import sys


def getargs():
    args = sys.argv
    if len(args) < 2:
        raise Exception("Expecting at least one argument")
    return (args[1], args[2:])


def drange(start, end, step=None):
    x = float(start)
    if step is None:
        s = 1.0
    else:
        s = step
    s = float(s)
    if s <=0:
        raise Exception("step <=0")
    while x < end:
        yield x
        x += s
    if x < end + s/2.0:
        yield end

def plot1(args):
    t = NPNTransistor(None, "", 1e-12, 25e-3, 100, 10)
    vb = 0.5
    ve = 0.02
    x = list(drange(-3, 3, 0.1))
    ie = [t.IE(vb-ve, vb-vc) for vc in x]
    ib = [t.IB(vb-ve, vb-vc) for vc in x]

    fig, ax = plt.subplots()  # Create a figure containing a single axes.
    plt.ion()
    plt.plot(x,ie)
    plt.plot(x,ib)
    plt.show()
    input()

def plot2(args):
    t = NPNTransistor(None, "", 1e-12, 25e-3, 100, 10)

    vc = 2
    ve = 0
    x = list(drange(-0.5,3.5, 0.01)    )
    ie = [t.IE(vb-ve, vb-vc) for vb in x]
    ib = [t.IB(vb-ve, vb-vc) for vb in x]
    fig, ax = plt.subplots()  # Create a figure containing a single axes.
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
    nw = Network()
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
    ax1.plot(x,y, color="black")
    ax2 = ax1.twinx()
    ax2.plot(x,z, color="green")
    fig.tight_layout()
    ax.plot(x, iy)
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
    s = 0.2
    x = 0
    while x < 10:
        res = ana.analyze(maxit=30, charges={"ca": ch })
        ica = res.get_current(c.p)
        ch += s * ica
        x += s
        xs.append(x)
        ys.append(ch)
    fig, (a1,a2) = plt.subplots(2)
    a1.plot(xs,ys, color="blue")
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
    elif cmd == "e":
        emitter(args)
    elif cmd == "c":
        capa(args)
    else:
        raise Exception("unknown commnd: {0}".format(cmd))

main()

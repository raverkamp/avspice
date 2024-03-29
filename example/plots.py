import math
import argparse
import sys

import matplotlib as mp
import matplotlib.pyplot as plt
import matplotlib.ticker

from avspice import *
from avspice.util import *


from avspice.ncomponents import NNPNTransistor


def plot1(args):
    # das passt alles nicht
    # oder doch: Basis Spannug sollte über der vom Kollektor liegen
    t = NNPNTransistor(1e-12, 25e-3, 100, 10, -40, 40)
    vbl = [0, 0.1, 0.2, 0.3, 0.4]
    k = math.ceil(math.sqrt(len(vbl)))
    (f, plts) = plt.subplots(k, k)
    ve = 0
    x = list(drange(0, 0.3, 0.01))
    i = 0
    for vb in vbl:
        ax = plts[i // k][i % k]
        ie = [t.IE(vb - ve, vb - vc) for vc in x]
        ib = [t.IB(vb - ve, vb - vc) for vc in x]
        te = "npn transistor, vb={0}, ve={1}".format(vb, ve)
        ax.set_title(te)
        ax.set_ylabel("current")
        ax.set_xlabel("vc")
        ax.plot(x, ie, label="ie")
        ax.plot(x, ib, label="ib")
        ax.legend()
        i += 1

    plt.ion()
    plt.show()
    input()


def plot2(args):
    t = NNPNTransistor(1e-12, 25e-3, 100, 10, -40, 40)

    vc = 2
    ve = 0
    x = list(drange(-0.5, 0.8, 0.01))
    ie = [t.IE(vb - ve, vb - vc) for vb in x]
    ib = [t.IB(vb - ve, vb - vc) for vb in x]
    _, _ = plt.subplots()  # Create a figure containing a single axes.
    plt.plot(x, ie, color="black")
    plt.plot(x, ib, color="green")
    plt.show()
    input()


def plot3(args):
    t = NNPNTransistor(1e-12, 25e-3, 100, 10, -40, 40)
    x = list(drange(-0.5, 3.5, 0.01))
    vc = args.cpot
    ve = 0
    ie = [t.IE(vb - ve, vb - vc) for vb in x]
    ib = [t.IB(vb - ve, vb - vc) for vb in x]
    ic = [t.IC(vb - ve, vb - vc) for vb in x]
    plt.plot(x, ie, color="black")
    plt.plot(x, ib, color="green")
    plt.plot(x, ic, color="blue")
    plt.show()
    input()


def plot4(args):
    sol = None

    tt = NPNTransistor("", 1e-12, 25e-3, 100, 10)

    def generate(er):
        x = list(drange(0, 6, 0.01))
        y = []
        z = []
        net = Circuit()
        net.addV("vc", 2, "v", "0")
        net.addV("vb", Variable("vb"), "vb", "0")
        net.addR("rb", 10e3, "vb", "B")

        if er:
            net.addR("re", 100, "E", "0")
            net.add_component("T1", tt, ("B", "v", "E"))
        else:
            net.addR("re", 100, "v", "C")
            net.add_component("T1", tt, ("B", "C", "0"))

        ana = Analysis(net)
        sol = None
        for v in x:
            res = ana.analyze(maxit=30, start_solution_vec=sol, variables={"vb": v})
            if isinstance(res, str):
                print("no covergence at: {0}".format(v))
                y.append(None)
                z.append(None)
                sol = None
            else:
                y.append(res.get_current("T1.C"))
                z.append(res.get_current("T1.B"))
                sol = res.solution_vec
        return (x, y, z)

    fig, (ax, bx) = plt.subplots(2)
    (x, y, z) = generate(True)
    ax.set_title("Emitter resistor")
    ax.plot(x, y, color="black", label="I(C)")
    ax.legend(loc=0)
    ax.set_ylabel("I(C)")
    ax2 = ax.twinx()
    ax2.plot(x, z, color="green", label="I(B)")
    ax2.legend(loc=1)
    ax2.set_ylabel("I(B)")

    (x, y, z) = generate(False)
    bx.plot(x, y, color="black", label="I(C)")
    bx.legend(loc=0)
    bx.set_ylabel("I(C)")
    bx.set_title("Collector resistor")
    bx2 = bx.twinx()
    bx2.plot(x, z, color="green", label="I(B)")
    bx2.legend(loc=1)
    bx2.set_ylabel("I(B)")

    fig.tight_layout()
    plt.show()
    # input()


def plot5(args):
    tt = NPNTransistor("X", 1e-12, 25e-3, 100, 10)

    net = Circuit()
    net.addV("vc", 5, "vcc", "0")
    net.addR("rc", 1e2, "vcc", "t1c")
    net.addR("rb", Variable("rb"), "vcc", "rbn")
    #    net.addR("rb", 1e4, "vcc", "rbp")
    net.add_component("T1", tt, ("rbn", "t1c", "e"))
    net.addR("re", 1, "e", "0")

    ana = Analysis(net)
    sol = None
    lrb = []
    lvb = []
    lcb = []
    for vrb in drange(1e3, 1e6, 1000):
        res = ana.analyze(maxit=30, variables={"rb": vrb})
        if isinstance(res, str):
            print("no covergence at: {0}".format(vrb))
            lrb.append(None)
            lvb.append(None)
            lcb.append(None)
        else:
            lrb.append(vrb)
            lvb.append(res.get_voltage("T1.B"))
            lcb.append(res.get_current("T1.B"))
    fig, (ax1, ax2) = plt.subplots(2)
    ax1.plot(lrb, lvb, color="black")
    ax2.plot(lrb, lcb, color="black")
    fig.tight_layout()
    plt.show()
    # input()


def emitter(args):
    tt = NPNTransistor("X", 1e-12, 25e-3, 100, 10)
    nw = Network()
    net = Circuit()
    net.addV("vc", 5, "vc", "0")
    net.addR("r1", 2000, "vc", "t1b")
    net.addR("r2", 2000, "t1b", "0")

    net.addV("vb", Variable("vb"), "vb", "0")
    net.addR("rc", 1e3, "vc", "t1c")
    net.addR("rb", 1e5, "vb", "t1b")

    net.add_component("t1", tt, ("t1b", "t1c", "0"))

    y = []
    z = []
    ana = Analysis(net)
    sol = None
    x = list(drange(0, 0.1, 0.001))
    for v in x:
        res = ana.analyze(maxit=30, start_solution_vec=sol, variables={"vb": v})
        if isinstance(res, str):
            print("no covergence at: {0}".format(v))
            y.append(None)
            sol = None
        else:
            y.append(res.get_voltage("t1.C"))
            z.append(res.get_voltage("t1.B"))
            sol = res.solution_vec
    fig, (a1, a2) = plt.subplots(2)
    a1.plot(x, y, color="blue")
    a2.plot(x, z, color="blue")
    plt.show()


def saw1(args):
    net = Circuit()
    r = 100000
    capa = 10e-6
    net.addSawV("vc", 1, 1, "v", "0")
    net.addR("r1", r, "v", "ca")
    net.addCapa("ca", capa, "ca", "0")

    ana = Analysis(net)
    res = ana.transient(9, 0.0001, capa_voltages={"ca": 0})

    (f, p1) = plt.subplots(1)
    p1.set_title("Voltage at capacitor, R={0}, CAPA={1}".format(r, capa))
    p1.plot(res.get_time(), res.get_voltage("ca.p"))
    plt.show()


def emitterschaltung(args):
    #   https://www.elektronik-kompendium.de/sites/slt/0204302.htm
    tt = NPNTransistor("", 1e-12, 25e-3, 100, 10)
    net = Circuit()
    net.addV("vc", 10, "vcc", "0")
    net.addSawV("ve", 0.1, 10, "ve", "0")

    net.add_component("tr", tt, ("B", "C", "0"))

    net.addCapa("ce", 100e-6, "ve", "B")
    # ce = net.addR("ce", 1)
    net.addCapa("ca", 1e-6, "C", "last")

    net.addR("r1", 10e3, "vcc", "B")
    net.addR("r2", 0.5e3, "B", "0")

    net.addR("rc", 1000, "vcc", "C")

    rl = net.addR("rl", 100, "last", "0")

    ana = Analysis(net)
    res = ana.transient(0.4, 0.0001, capa_voltages={"ca": 0, "ce": 0})

    (fig, (p1, p2, p3, p4)) = plt.subplots(4)
    p1.plot(res.get_time(), res.get_current("tr.B"))
    p1.set_title("current base")
    p2.plot(res.get_time(), res.get_voltage("tr.B"))
    p2.set_title("voltage base")

    p3.plot(res.get_time(), res.get_voltage("rl.p"))
    p3.set_title("voltage out")
    p4.plot(res.get_time(), res.get_current("rl.p"))
    p4.set_title("current out")
    fig.tight_layout()

    plt.show()


def rlc(args):
    net = Circuit()
    i0 = 0.1
    rv = 100
    indu = 1
    capa = 10e-6

    net.addR("r1", rv, "1", "2")
    net.addCapa("ca", capa, "2", "0")
    net.addInduc("ind", indu, "0", "1")

    ana = Analysis(net)
    res = ana.transient(0.1, 0.0001, induc_currents={"ind": 1}, capa_voltages={"ca": 0})

    (fig, ax) = plt.subplots(1)
    ax.plot(res.get_time(), res.get_current("r1.p"), label="current", color="blue")
    f = 1 / (math.sqrt(capa * indu) * 2 * math.pi)
    ax.set_title(f"RLC r={rv}, c={capa}, l={indu}, freq={f}")

    formatterx = matplotlib.ticker.EngFormatter("s")
    formattery = matplotlib.ticker.EngFormatter("A")
    ax.xaxis.set_major_formatter(formatterx)
    ax.xaxis.set_minor_formatter(formatterx)
    ax.yaxis.set_major_formatter(formattery)
    ax.yaxis.set_minor_formatter(formattery)

    ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.plot(
        res.get_time(),
        res.get_voltage("ind.p") - res.get_voltage("ind.n"),
        label="voltage across inductor",
        color="red",
    )
    formattery2 = matplotlib.ticker.EngFormatter("V")
    ax2.yaxis.set_major_formatter(formattery2)
    ax2.yaxis.set_minor_formatter(formattery2)
    fig.legend()
    plt.show()


def tricky(args):
    # loading a capacitor and an switching a transistor base
    # when loaded
    capa = 100e-6
    net = Circuit()
    tt = NPNTransistor("", 1e-12, 25e-3, 100, 10)
    net.add_component("t", tt, ("B", "C", "0"))
    net.addV("v", 9, "v", "0")
    net.addR("r1", 100, "v", "C")
    net.addR("r2", 50e3, "C", "B")
    net.addCapa("ca", capa, "B", "0")

    ana = Analysis(net)
    res = ana.transient(0.6, 0.0001, capa_voltages={"ca": 0.0})
    time = res.get_time()

    (fig, (a1, a2, a3)) = plt.subplots(3)
    a1.set_title("curr(t.E)")
    a1.plot(time, res.get_current("t.E"))
    a2.set_title("curr(ca)")
    a2.plot(time, res.get_current("ca.p"))
    a3.set_title("volts(B)")
    a3.plot(time, res.get_voltage("t.B"))
    fig.legend()
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(required=True)

    parser_1 = subparsers.add_parser("1")
    parser_1.set_defaults(func=plot1)

    parser_2 = subparsers.add_parser("2")
    parser_2.set_defaults(func=plot2)

    parser_3 = subparsers.add_parser("3")
    parser_3.set_defaults(func=plot3)

    parser_3.add_argument("cpot", type=float)
    parser_3.add_argument("cutoff", type=float)

    parser_4 = subparsers.add_parser("4")
    parser_4.set_defaults(func=plot4)

    parser_5 = subparsers.add_parser("5")
    parser_5.set_defaults(func=plot5)

    parser_e = subparsers.add_parser("e")
    parser_e.set_defaults(func=emitter)

    parser_saw = subparsers.add_parser("saw")
    parser_saw.set_defaults(func=saw1)

    parser_emitter = subparsers.add_parser("emitter")
    parser_emitter.set_defaults(func=emitterschaltung)

    parser_rlc = subparsers.add_parser("rlc")
    parser_rlc.set_defaults(func=rlc)

    parser_tricky = subparsers.add_parser("tricky")
    parser_tricky.set_defaults(func=tricky)

    args = parser.parse_args()
    args.func(args)


main()

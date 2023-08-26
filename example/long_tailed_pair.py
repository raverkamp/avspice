"""long tailed pair"""

import argparse
import matplotlib.pyplot as plt

from avspice import (
    Diode,
    NPNTransistor,
    PNPTransistor,
    Circuit,
    SubCircuit,
    Variable,
    Analysis,
    ZDiode,
)
from avspice.util import drange


def create_current_mirror_npn(model):
    assert isinstance(model, NPNTransistor)
    sc = SubCircuit(("inc", "outc", "00"))
    sc.add("Tin", model, ("inc", "inc", "00"))
    sc.add("Tout", model, ("inc", "oiutc", "00"))
    return sc


def create_current_mirror_pnp(model):
    assert isinstance(model, PNPTransistor)
    sc = SubCircuit(("inc", "outc", "top"))
    sc.add("Tin", model, ("inc", "inc", "top"))
    sc.add("Tout", model, ("inc", "outc", "top"))
    return sc


def create_current_source_npn(tmodel: NPNTransistor, zmodel: ZDiode, r: float):
    assert isinstance(tmodel, NPNTransistor)
    assert isinstance(zmodel, ZDiode)

    sc = SubCircuit(("vc", "top", "bot"))
    sc.addR("RB", 1e3, "vc", "B")
    sc.add("Q", tmodel, ["B", "top", "Rout"])
    sc.addR("RBot", r, "Rout", "bot")
    sc.addV("V5", 5, "B", "bot")
    #    sc.add("D", zmodel, ["bot","B"])
    return sc


def cmd_current_source(args):
    r = (5 - 0.65) / args.cu
    rcin = Variable("r", 0)
    net = Circuit()
    net.addV("V", 10, "V", "0")
    net.addR("R", rcin, "V", "top")
    t1 = NPNTransistor("NPn", 1e-12, 25e-3, 100, 10)
    z1 = ZDiode("Z5", 5, 1e-8, 25e-3)
    a = create_current_source_npn(t1, z1, r)
    net.add("CS", a, ["V", "top", "0"])

    ana = Analysis(net)
    res = ana.analyze(variables={"r": 10}, maxit=args.maxit)
    if isinstance(res, str):
        print(res)
    else:
        res.display()


def cmd_plot_mirror(args):
    rcin = Variable("rcin", 0)
    net = Circuit()

    net.addV("V", 10, "pV", "0")
    ts = PNPTransistor("PNP", 1e-12, 25e-3, 100, 10)
    q = create_current_mirror_pnp(ts)

    net.add("MIR", q, ["in", "out", "pV"])
    net.addR("RC", rcin, "in", "0")

    net.addR("RCout", 100, "out", "0")

    x = []
    y = []
    z = []

    ana = Analysis(net)
    for r in drange(100, 1000, 1):
        x.append(r)
        res = ana.analyze(variables={"rcin": r}, maxit=args.maxit)
        if isinstance(res, str):
            y.append(None)
            z.append(None)
        else:
            y.append(res.get_current("RCout.p"))
            z.append(res.get_current("RC.p"))

    fig, (ax1, ax2) = plt.subplots(2)
    ax1.plot(x, y)
    ax2.plot(x, z)
    fig.tight_layout()
    plt.show()


def cmd_plot_mirror2(args):
    rcin = Variable("rcin", 0)
    net = Circuit()

    net.addV("V", 10, "pV", "0")

    ts = NPNTransistor("NPn", 1e-12, 25e-3, 100, 10)

    net.addR("RC", rcin, "pV", "t1C")
    net.addR("RC2", 100, "pV", "t2C")

    net.add("T1", ts, ("t1C", "t1C", "0"))

    net.add("T2", ts, ("t1C", "t2C", "0"))

    x = []
    y = []
    z = []
    ib1 = []
    ib2 = []
    vc1 = []
    vc2 = []
    ana = Analysis(net)
    for r in drange(100, 1000, 1):
        x.append(r)
        res = ana.analyze(variables={"rcin": r}, maxit=args.maxit)
        if isinstance(res, str):
            y.append(None)
            z.append(None)
            ib1.append(None)
            ib2.append(None)
        else:
            y.append(res.get_current("T1.C"))
            z.append(res.get_current("T2.C"))
            ib1.append(res.get_current("T1.B"))
            ib2.append(res.get_current("T2.B"))

            vc1.append(res.get_voltage("T1.C"))
            vc2.append(res.get_voltage("T2.C"))

    fig, ((ax1, ax2), (bx1, bx2), (cx1, cx2)) = plt.subplots(3, 2)
    ax1.plot(x, y)
    ax2.plot(x, z)
    bx1.plot(x, ib1)
    bx2.plot(x, ib2)

    cx1.plot(x, vc1)
    cx2.plot(x, vc2)
    fig.tight_layout()

    plt.show()


def create_circuit(rc, re, beta, cu):
    dv = 5

    tt = NPNTransistor("NPN", 1e-12, 25e-3, beta, 10)

    ts = PNPTransistor("PNP", 1e-12, 25e-3, 100, 10)

    net = Circuit()

    net.addV("V1", dv, "pVCC", "0")
    net.addV("V2", dv, "0", "nVCC")

    if rc > 0:
        net.addR("RC1", rc, "pVCC", "t1c")
        net.addR("RC2", rc, "pVCC", "t2c")
    else:
        q = create_current_mirror_pnp(ts)
        net.add("MIR", q, ["t1c", "t2c", "pVCC"])

    if re > 0:
        net.addR("RE", re, "e", "nVCC")
    else:
        net.addC("CS", cu, "e", "nVCC")
    #        z = ZDiode("Z5", 5, 1e-8, 25e-3)
    #        cus = create_current_source_npn(tt, z, 5-0.3/cu)
    #        net.add("CS",cus, ["pVCC","e", "nVCC"])

    net.add("t1", tt, ("ip", "t1c", "e"))
    net.add("t2", tt, ("in", "t2c", "e"))

    vp = Variable("vn", 0)
    vn = Variable("vp", 0)

    net.addV("VP", vp, "ip", "0")
    net.addV("VN", vn, "in", "0")

    return net


def cmd_analysis(args):
    if (args.re >= 0 and args.cu >= 0) or (args.re <= 0 and args.cu <= 0):
        raise Exception("re or cu must be given")

    net = create_circuit(rc=args.rc, re=args.re, beta=args.beta, cu=args.cu)

    signal_p = args.v + args.dv / 2
    signal_n = args.v - args.dv / 2

    ana = Analysis(net)
    res = ana.analyze(
        variables={"vp": signal_p, "vn": signal_n},
        maxit=args.maxit,
        verbose=args.verbose,
        nrandom=10000,
    )
    if isinstance(res, str):
        print(res)
    else:
        res.display()
        print(
            (
                "OUT:",
                res.get_voltage("t1.C"),
                res.get_voltage("t2.C"),
                res.get_voltage("t1.C") - res.get_voltage("t2.C"),
                (res.get_voltage("t1.C") - res.get_voltage("t2.C")) / args.dv,
            )
        )


def cmd_plot(args):

    if (args.re >= 0 and args.cu >= 0) or (args.re >= 0 and args.cu >= 0):
        raise Exception("re or cu must be given")

    net = create_circuit(rc=args.rc, re=args.re, beta=args.beta, cu=-args.cu)

    x = []
    y = []
    z = []
    dv = 0.0001
    ana = Analysis(net)
    solution_vec = None
    for v in drange(1, 4, 0.01):
        x.append(v)

        signal_p = v + dv / 2
        signal_n = v - dv / 2
        res = ana.analyze(
            start_solution_vec=solution_vec,
            variables={"vp": signal_p, "vn": signal_n},
            maxit=args.maxit,
            nrandom=100,
        )
        if isinstance(res, str):
            y.append(None)
            z.append(None)
            solution_vec = None
        else:
            y.append((res.get_voltage("t1.C") - res.get_voltage("t2.C")) / dv)
            z.append(res.get_voltage("t1.C"))
            solution_vec = res.solution_vec

    fig, (ax1, ax2) = plt.subplots(2)
    fig.suptitle(
        f"beta={args.beta}, rc={args.rc} "
        + ("re={args.re}" if args.re > 0 else f"cu={args.cu}")
    )
    ax1.plot(x, y)
    ax2.plot(x, z)
    ax1.set_title(f"gain")
    ax1.set_xlabel("v(input)")
    ax1.set_ylabel("gain")
    ax2.set_title("t1.C")
    ax2.set_xlabel("v(input)")
    ax2.set_ylabel("t1.C")
    fig.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(required=True)

    parser_comp = subparsers.add_parser("comp")
    parser_comp.set_defaults(func=cmd_analysis)
    parser_comp.add_argument("v", type=float, default=0)
    parser_comp.add_argument("dv", type=float, default=0)
    parser_comp.add_argument("-rc", type=float, default=1000)
    parser_comp.add_argument("-re", type=float, default=-1)
    parser_comp.add_argument("-cu", type=float, default=-1)
    parser_comp.add_argument("-beta", type=float, default=100)
    parser_comp.add_argument("-maxit", type=int, default=20)
    parser_comp.add_argument("-verbose", type=bool, default=False)

    parser_plot = subparsers.add_parser("plot")
    parser_plot.set_defaults(func=cmd_plot)
    parser_plot.add_argument("-rc", type=float, default=1000)
    parser_plot.add_argument("-re", type=float, default=-1)
    parser_plot.add_argument("-cu", type=float, default=-1)
    parser_plot.add_argument("-beta", type=float, default=100)
    parser_plot.add_argument("-maxit", type=int, default=20)

    parser_mirror = subparsers.add_parser("mirror")
    parser_mirror.set_defaults(func=cmd_plot_mirror)
    parser_mirror.add_argument("-maxit", type=int, default=20)

    parserx = subparsers.add_parser("mirror2")
    parserx.set_defaults(func=cmd_plot_mirror2)
    parserx.add_argument("-maxit", type=int, default=20)

    pa = subparsers.add_parser("cs")
    pa.set_defaults(func=cmd_current_source)
    pa.add_argument("cu", type=float)
    pa.add_argument("-maxit", type=int, default=20)

    args = parser.parse_args()
    args.func(args)


main()

"""long tailed pair"""

import argparse
import matplotlib.pyplot as plt

from avspice import Diode, NPNTransistor, Circuit, SubCircuit, Variable, Analysis
from avspice.util import drange


def create_circuit(rc, re, beta):
    dv = 5

    tt = NPNTransistor("tt", 1e-12, 25e-3, beta, 10)

    net = Circuit()

    net.addV("V1", dv, "pVCC", "0")
    net.addV("V2", dv, "0", "nVCC")

    net.addR("RC1", rc, "pVCC", "t1c")
    net.addR("RC2", rc, "pVCC", "t2c")

    net.addR("RE", re, "e", "nVCC")

    net.add("t1", tt, ("ip", "t1c", "e"))
    net.add("t2", tt, ("in", "t2c", "e"))

    vp = Variable("vn", 0)
    vn = Variable("vp", 0)

    net.addV("VP", vp, "ip", "0")
    net.addV("VN", vn, "in", "0")

    return net


def cmd_analysis(args):
    net = create_circuit(rc=args.rc, re=args.re, beta=args.beta)

    signal_p = args.v + args.dv / 2
    signal_n = args.v - args.dv / 2

    ana = Analysis(net)
    res = ana.analyze(variables={"vp": signal_p, "vn": signal_n}, maxit=args.maxit)
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
            )
        )


def cmd_plot(args):
    beta = args.beta
    rc = args.rc
    re = args.re
    net = create_circuit(rc=rc, re=re, beta=beta)

    x = []
    y = []
    z = []
    dv = 0.001
    ana = Analysis(net)
    for v in drange(1, 4, 0.01):
        x.append(v)

        signal_p = v + dv / 2
        signal_n = v - dv / 2
        res = ana.analyze(variables={"vp": signal_p, "vn": signal_n}, maxit=args.maxit)
        if isinstance(res, str):
            y.append(None)
            z.append(None)
        else:
            y.append((res.get_voltage("t1.C") - res.get_voltage("t2.C")) / dv)
            z.append(res.get_voltage("t1.C"))

    fig, (ax1, ax2) = plt.subplots(2)
    ax1.plot(x, y)
    ax2.plot(x, z)
    ax1.set_title(f"beta={beta}, rc={rc}, re={re}")
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(required=True)

    parser_comp = subparsers.add_parser("comp")
    parser_comp.set_defaults(func=cmd_analysis)
    parser_comp.add_argument("v", type=float, default=0)
    parser_comp.add_argument("dv", type=float, default=0)
    parser_comp.add_argument("-rc", type=float, default=1000)
    parser_comp.add_argument("-re", type=float, default=1e3)
    parser_comp.add_argument("-beta", type=float, default=100)
    parser_comp.add_argument("-maxit", type=int, default=20)

    parser_plot = subparsers.add_parser("plot")
    parser_plot.set_defaults(func=cmd_plot)
    parser_plot.add_argument("-rc", type=float, default=1000)
    parser_plot.add_argument("-re", type=float, default=1e3)
    parser_plot.add_argument("-beta", type=float, default=100)
    parser_plot.add_argument("-maxit", type=int, default=20)

    args = parser.parse_args()
    args.func(args)


main()

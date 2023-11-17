"""long tailed pair"""

import argparse
import matplotlib.pyplot as plt
import pprint as pp

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

from example_circuits import create_current_source_npn, create_current_mirror_pnp


def cmd_current_source(args):
    supply_voltage = args.supply_voltage
    current = args.current
    zdiode_voltage = args.zdiode_voltage

    rcin = Variable("r", 0)
    net = Circuit()
    net.addV("V", supply_voltage, "V", "0")
    net.addR("R", rcin, "V", "top")
    t1 = NPNTransistor("NPN", 1e-12, 25e-3, 100, 10)
    a = create_current_source_npn(t1, zdiode_voltage, current)
    net.add("CS", a, ["V", "top", "0"])

    ana = Analysis(net)

    x = []
    y = []
    z = []

    maxr = 1.01 * (supply_voltage / current)
    solution_vec = None

    for rl in drange(1, maxr, maxr / 1000):
        x.append(rl)
        res = ana.analyze(
            variables={"r": rl}, maxit=args.maxit, start_solution_vec=solution_vec
        )
        if isinstance(res, str):
            y.append(None)
            z.append(None)
            solution_vec = None
        else:
            y.append(res.get_current("R.p"))
            z.append((res.get_current("R.p") / current - 1) * 100)
            solution_vec = res.solution_vec
        if res.get_current("R.p") / current < 0.9:
            break

    fig, (ax1, ax2) = plt.subplots(2)
    ax1.plot(x, y)
    ax1.set_xlabel("RL")
    ax1.set_ylabel("Current")
    ax2.plot(x, z)
    ax2.set_xlabel("RL")
    ax2.set_ylabel("Error %")

    fig.suptitle(
        f"Current Source, target current={current}, supply voltage={supply_voltage}, zdiode voltage={zdiode_voltage}"
    )
    fig.tight_layout()
    plt.show()


def cmd_plot_mirror(args):
    cin = Variable("current", 0)
    net = Circuit()

    net.addV("V", 10, "pV", "0")
    ts = PNPTransistor("PNP", args.IS, 25e-3, args.beta, 10)
    q = create_current_mirror_pnp(ts)

    net.add("MIR", q, ["in", "out", "pV"])
    net.addC("CU", cin, "in", "0")

    net.addR("RCout", 10, "out", "0")

    x = []
    y = []
    z = []
    w = []

    ana = Analysis(net)
    for current in drange(0.01, 0.9, 0.01):
        x.append(current)
        res = ana.analyze(variables={"current": -current}, maxit=args.maxit)
        if isinstance(res, str):
            y.append(None)
            z.append(None)
            w.append(None)
        else:
            y.append(res.get_current("RCout.p"))
            z.append(current - res.get_current("RCout.p"))
            w.append(100 * (current - res.get_current("RCout.p")) / current)

    fig, (ax1, ax2, ax3) = plt.subplots(3)
    ax1.plot(x, y)
    ax1.set_xlabel("Input Current")
    ax1.set_ylabel("Output Current")
    ax2.plot(x, z)
    ax2.set_xlabel("Input Current")
    ax2.set_ylabel("Input- Output Current")
    ax3.plot(x, w)
    ax3.set_xlabel("Input Current")
    ax3.set_ylabel("% Error")
    fig.suptitle(f"Current Mirror beta={args.beta}, IS={args.IS}")
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
        # net.addC("CS", cu, "e", "nVCC")
        z = ZDiode("Z5", 5, 1e-8, 25e-3)
        cus = create_current_source_npn(tt, z, (5 - 0.65) / cu)
        net.add("CS", cus, ["pVCC", "e", "nVCC"])

    net.add("t1", tt, ("ip", "t1c", "e"))
    net.add("t2", tt, ("in", "t2c", "e"))

    vp = Variable("vn", 0)
    vn = Variable("vp", 0)

    net.addV("VP", vp, "ip", "0")
    net.addV("VN", vn, "in", "0")

    return net


def cmd_analysis(args):
    pp.pprint(args)
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
        nrandom=100,
        abstol=1e-20,
    )
    if isinstance(res, str):
        print(res)
    else:
        res.display()
        pp.pprint(
            {
                "V(t1.C)": res.get_voltage("t1.C"),
                "(V(t2.C)": res.get_voltage("t2.C"),
                "v1-v2": res.get_voltage("t1.C") - res.get_voltage("t2.C"),
                "gain": (res.get_voltage("t1.C") - res.get_voltage("t2.C")) / args.dv,
                "Current(t1.C)": res.get_current("t1.C"),
                "Current(t2.C)": res.get_current("t2.C"),
                "Current(t1.B)": res.get_current("t1.B"),
                "Current(t2.B)": res.get_current("t2.B"),
                "Current(CS)": res.get_current("CS/RBot.p"),
            }
        )


def cmd_plot(args):
    if (args.re >= 0 and args.cu >= 0) or (args.re >= 0 and args.cu >= 0):
        raise Exception("re or cu must be given")

    pp.pprint(args)
    net = create_circuit(rc=args.rc, re=args.re, beta=args.beta, cu=args.cu)

    x = []
    y = []
    z = []
    dv = args.dv
    ana = Analysis(net)
    solution_vec = None
    for v in drange(-4, 4, 0.01):
        x.append(v)

        signal_p = v + dv / 2
        signal_n = v - dv / 2
        res = ana.analyze(
            start_solution_vec=solution_vec,
            variables={"vp": signal_p, "vn": signal_n},
            maxit=args.maxit,
            nrandom=100,
        )
        # res.display()
        # break

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

    pa = subparsers.add_parser("comp")
    pa.set_defaults(func=cmd_analysis)
    pa.add_argument("v", type=float, default=0, help="voltage at inputs")
    pa.add_argument(
        "dv", type=float, default=0, help="voltage difference between inputs"
    )
    pa.add_argument(
        "-rc",
        type=float,
        default=1000,
        help="collector resistor, <0 means use currency mirror",
    )
    pa.add_argument("-re", type=float, default=-1, help="emitter resistance")
    pa.add_argument("-cu", type=float, default=-1, help="emitter current")
    pa.add_argument("-beta", type=float, default=100, help="beta of the transistors")
    pa.add_argument("-maxit", type=int, default=20, help="max iteration")
    pa.add_argument("-verbose", type=bool, default=False, help="verbose messaging")

    pa = subparsers.add_parser("plot")
    pa.set_defaults(func=cmd_plot)
    pa.add_argument("-rc", type=float, default=1000)
    pa.add_argument("-re", type=float, default=-1)
    pa.add_argument("-cu", type=float, default=-1)
    pa.add_argument("-beta", type=float, default=100)
    pa.add_argument("-maxit", type=int, default=20)
    pa.add_argument("-dv", type=float, default=0.01)

    pa = subparsers.add_parser("mirror", description="current mirror plot")
    pa.set_defaults(func=cmd_plot_mirror)
    pa.add_argument("-maxit", type=int, default=20)
    pa.add_argument("-beta", type=float, default=100)
    pa.add_argument("-IS", type=float, default=1e-12)

    pa = subparsers.add_parser("current_source", description="current source")
    pa.set_defaults(func=cmd_current_source)
    pa.add_argument(
        "current", type=float, help="target current", metavar="target_current"
    )
    pa.add_argument(
        "-supply_voltage",
        type=float,
        help="supply voltage",
        metavar="supply_voltage",
        default=10,
    )
    pa.add_argument(
        "-zdiode_voltage",
        type=float,
        help="zdiode voltage",
        metavar="zdiode_voltage",
        default=5,
    )
    pa.add_argument("-maxit", type=int, default=20)

    args = parser.parse_args()
    args.func(args)


main()

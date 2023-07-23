"""long tailed pair"""

import argparse
import matplotlib.pyplot as plt

from avspice import Diode, NPNTransistor, Circuit, SubCircuit, Variable, Analysis
from avspice.util import drange


def mk(args):
    beta = 100

    dv = 5
    rc = 1e3
    re = 10e3

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

    ana = Analysis(net)
    res = ana.analyze(variables={"vp": args.vp, "vn": args.vn})
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


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(required=True)

    parser_comp = subparsers.add_parser("comp")
    parser_comp.set_defaults(func=mk)
    parser_comp.add_argument("vp", type=float, default=0)
    parser_comp.add_argument("vn", type=float, default=0)

    args = parser.parse_args()
    args.func(args)


main()

""" resistor transistor logic"""
import argparse
from avspice import *


tnpn =  NPNTransistor("TT", 1e-12, 25e-3,100, 10)


def create_not_gate(*,ri, rc):
    """  create  a NOT gate """
    sc =  SubCircuit(("vc", "0", "i", "o"))
    sc.add_component("T", tnpn, ("b","o","0"))
    sc.addR("rc", rc, "vc","o")
    sc.addR("ri", ri, "i", "b")
    return sc


def not1(args):
    net = Circuit()
    net.addV("V", 5, "vc","0")

    vi = Variable("vi")
    ri = 1e4
    rc = 1e3
    ro = ri/3

    net.addV("vi", vi, "i", "0")

    ng = create_not_gate(ri=ri, rc=rc)

    net.add_subcircuit("n1", ng, ("vc","0","i", "o1"))

    net.addR("ro", ro, "o1", "0")

    ana = Analysis(net)

    res= ana.analyze(variables={"vi":args.vi})
    print(f"----   not1 vi={args.vi} -----")
    res.display()


def not2(args):
    net = Circuit()

    vi = Variable("vi")

    ri = 1e4
    rc = 1e3
    ro = ri/3

    ng = create_not_gate(ri=ri, rc=rc)

    net.addV("V", 5, "v","0")

    net.addV("vi", vi, "i", "0")

    net.add_subcircuit("n1", ng, ("v","0","i", "o1"))
    net.add_subcircuit("n2", ng, ("v","0","o1", "o2"))

    net.addR("ro2", ro, "o2", "0")


    ana = Analysis(net)

    res= ana.analyze(variables={"vi": args.vi})
    print(f"----   not2 vi={args.vi} -----")
    res.display()

def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(required=True)

    p_not1 = subparsers.add_parser('not1')
    p_not1.set_defaults(func=not1)
    p_not1.add_argument("vi", type=float)

    p_not2d = subparsers.add_parser('not2')
    p_not2d.set_defaults(func=not2)
    p_not2d.add_argument("vi", type=float)


    args = parser.parse_args()
    args.func(args)
main()

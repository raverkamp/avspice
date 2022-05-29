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

def create_nor2_gate(*, ri, rc):
    sc =  SubCircuit(("vc", "0", "i1","i2", "o"))

    sc.addR("rc", rc, "vc","o")

    sc.addR("ri1", ri, "i1", "b1")
    sc.add_component("T1", tnpn, ("b1","o","0"))

    sc.addR("ri2", ri, "i2", "b2")
    sc.add_component("T2", tnpn, ("b2","o","0"))
    return sc

def create_or2_gate(*, ri, rc):
    sc =  SubCircuit(("v", "0", "i1","i2", "o"))

    nor2 = create_nor2_gate(ri=ri,rc=rc)
    not1 = create_not_gate(ri=ri, rc=rc)

    sc.add_subcircuit("nor", nor2, ("v","0","i1","i2", "o1"))
    sc.add_subcircuit("not", not1, ("v","0","o1","o"))

    return sc

def create_or4_gate(*, ri, rc):
    sc =  SubCircuit(("v", "0", "i1","i2", "i3", "i4", "o"))

    or2 = create_or2_gate(ri=ri,rc=rc)

    sc.add_subcircuit("or_i1", or2, ("v","0","i1","i2", "o1"))
    sc.add_subcircuit("or_i2", or2, ("v","0","i3","i4", "o2"))

    sc.add_subcircuit("or", or2, ("v","0", "o1", "o2", "o"))

    return sc


def cmd_not(args):
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


def cmd_not_not(args):
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

def cmd_nor2(args):
    net = Circuit()

    vi1 = Variable("vi1")
    vi2 = Variable("vi2")

    ri = 1e4
    rc = 1e3
    ro = ri/3

    ng = create_nor2_gate(ri=ri, rc=rc)

    net.addV("V", 5, "v","0")

    net.addV("vi1", vi1, "i1", "0")
    net.addV("vi2", vi2, "i2", "0")

    net.add_subcircuit("n1", ng, ("v","0","i1","i2", "o"))

    net.addR("ro", ro, "o", "0")


    ana = Analysis(net)

    res= ana.analyze(variables={"vi1": args.vi1, "vi2": args.vi2})
    print(f"----   nor2 vi={args.vi1} {args.vi2} -----")
    res.display()

def cmd_or2(args):
    net = Circuit()

    vi1 = Variable("vi1")
    vi2 = Variable("vi2")

    ri = 1e4
    rc = 1e3
    ro = ri/3

    or2 = create_or2_gate(ri=ri, rc=rc)

    net.addV("V", 5, "v","0")

    net.addV("vi1", vi1, "i1", "0")
    net.addV("vi2", vi2, "i2", "0")

    net.add_subcircuit("or", or2, ("v","0","i1","i2", "o"))
    net.addR("ro", ro, "o", "0")

    ana = Analysis(net)

    res= ana.analyze(variables={"vi1": args.vi1, "vi2": args.vi2})
    print(f"----   or2 vi={args.vi1} {args.vi2} -----")
    res.display()


def cmd_or4(args):
    net = Circuit()

    vi1 = Variable("vi1")
    vi2 = Variable("vi2")
    vi3 = Variable("vi3")
    vi4 = Variable("vi4")

    ri = 1e4
    rc = 1e3
    ro = ri/3

    or4 = create_or4_gate(ri=ri, rc=rc)

    net.addV("V", 5, "v","0")

    net.addV("vi1", vi1, "i1", "0")
    net.addV("vi2", vi2, "i2", "0")
    net.addV("vi3", vi3, "i3", "0")
    net.addV("vi4", vi4, "i4", "0")

    net.add_subcircuit("or", or4, ("v","0","i1","i2", "i3", "i4", "o"))
    net.addR("ro", ro, "o", "0")

    ana = Analysis(net)

    res= ana.analyze(variables={"vi1": args.vi1, "vi2": args.vi2, "vi3": args.vi3, "vi4": args.vi4})
    print(f"----   or4 vi={args.vi1} {args.vi2} -----")
    res.display()
    o =  res.get_voltage("ro.p")
    print(f"Outout: {o}")



def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(required=True)

    p_not = subparsers.add_parser('not')
    p_not.set_defaults(func=cmd_not)
    p_not.add_argument("vi", type=float)

    p_not_not = subparsers.add_parser('not_not')
    p_not_not.set_defaults(func=cmd_not_not)
    p_not_not.add_argument("vi", type=float)

    p_nor2 = subparsers.add_parser('nor2')
    p_nor2.set_defaults(func=cmd_nor2)
    p_nor2.add_argument("vi1", type=float)
    p_nor2.add_argument("vi2", type=float)

    p_or2 = subparsers.add_parser('or2')
    p_or2.set_defaults(func=cmd_or2)
    p_or2.add_argument("vi1", type=float)
    p_or2.add_argument("vi2", type=float)

    p_or4 = subparsers.add_parser('or4')
    p_or4.set_defaults(func=cmd_or4)
    p_or4.add_argument("vi1", type=float)
    p_or4.add_argument("vi2", type=float)
    p_or4.add_argument("vi3", type=float)
    p_or4.add_argument("vi4", type=float)


    args = parser.parse_args()
    args.func(args)

main()

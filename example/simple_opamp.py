"""simple opamp"""
import argparse
import matplotlib.pyplot as plt

from avspice import (
    SubCircuit,
    Circuit,
    Analysis,
    Variable,
    NPNTransistor,
    LinearVoltageControlledVoltageSource,
    Result,
)

from avspice.util import drange


def create_opamp(gain):
    s = SubCircuit(("VIN_p", "VIN_n", "VOUT", "zero"))

    s.addR("RIN", 1e6, "VIN_p", "VIN_n")

    c = LinearVoltageControlledVoltageSource(f"g_{gain}", gain)

    s.add_component("VGAIN", c, ["VIN_p", "VIN_n", "VOUT1", "zero"])
    s.addR("R1", 1e3, "VOUT1", "CIN1")
    s.addCapa("C1", 10e-6, "CIN1", "zero")

    c2 = LinearVoltageControlledVoltageSource("g1", 1)

    s.add_component("VBUFFER", c2, ["CIN1", "zero", "O1", "zero"])
    s.addR("ROUT", 100, "O1", "VOUT")
    return s


def cmd_gain1(args):
    net = Circuit()
    net.addV("VIN", 2, "INPUT", "0")
    s = create_opamp(3)
    net.add("OPAMP", s, ["INPUT", "0", "OUT", "0"])
    net.addR("RL", 1e3, "OUT", "0")

    ana = Analysis(net)
    res = ana.analyze()
    assert isinstance(res, Result)
    res.display()


def cmd_feedback(args):
    net = Circuit()
    dv = 3
    net.addV("VIN", dv, "INPUT", "0")
    s = create_opamp(1000)
    net.add("OPAMP", s, ["INPUT", "OUT", "OUT", "0"])
    net.addR("RL", 100, "OUT", "0")

    ana = Analysis(net)
    res = ana.analyze()
    assert isinstance(res, Result)
    res.display()


def main():
    parser = argparse.ArgumentParser(
        prog="Simple OpAmp Model",
        description="""example for a simple opamp""",
    )
    subparsers = parser.add_subparsers(required=True)

    p_gain1 = subparsers.add_parser("gain")
    p_gain1.set_defaults(func=cmd_gain1)

    p_feedback = subparsers.add_parser("feedback")
    p_feedback.set_defaults(func=cmd_feedback)

    args = parser.parse_args()
    args.func(args)


main()

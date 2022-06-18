"""a zenerdiode as subcircuit, current cruve and an example circuit"""

import argparse
import matplotlib.pyplot as plt

from avspice import Diode, NPNTransistor, Circuit, SubCircuit, Variable, Analysis
from avspice.util import drange

def zener(name, v):
    d =Diode("D", 1e-8, 25e-3)
    cutoff = 40
    d =Diode(name, 1e-8, 25e-3, lcut_off=-cutoff, rcut_off=cutoff)
    sc =  SubCircuit(("p", "n"))
    sc.add_component("Df", d, ("p", "n"))
    sc.add_component("Dr", d, ("m", "p"))
    sc.addV("v",v, "n", "m")
    return sc

def kennlinie(_):
    net = Circuit()
    v = Variable("v")
    net.addV("V", v, "VCC", "0")
    z = zener("Z5", 5)
    net.add_subcircuit("Z", z, ("VCC", "0"))
    ana = Analysis(net)

    x = []
    y = []
    sol_vec = None
    for v in drange (-5.5, 0.5,0.01):
        res = ana.analyze(variables={"v": v}, start_solution_vec=sol_vec)
        if not isinstance(res, str):
            x.append(v)
            y.append(res.get_current("V.p"))
            sol_vec = res.solution_vec
    _, _ = plt.subplots()  # Create a figure containing a single axes.
    plt.plot(x,y)
    plt.show()

def voltage_control(_):
    net = Circuit()
    v = Variable("v", 10)
    net.addV("V", v, "VCC", "0")

    z = zener("Z5", 5)
    net.add_subcircuit("Z", z, ("0", "m"))

    #    rv = Variable("rv", 10e3)
    net.addR("rm", 110e3, "VCC", "m")

    tt = NPNTransistor("tt", 1e-12, 25e-3, 100, 10)

    net.add_component("T1", tt, ("m","VCC", "o1"))
    net.add_component("T2", tt, ("o1", "VCC", "o"))
    net.addR("RL", 100, "o", "0")
    ana = Analysis(net)

    x = []
    y = []
    sol_vec = None
    for v in drange (0, 10,0.01):
        res = ana.analyze(variables={"v": v}, start_solution_vec=sol_vec)
        if not isinstance(res, str):
            x.append(v)
            y.append(res.get_voltage("RL.p"))
            sol_vec = res.solution_vec
    _, _ = plt.subplots()  # Create a figure containing a single axes.
    plt.plot(x,y)
    plt.show()




def main():

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(required=True)

    parser_kennlinie = subparsers.add_parser('k')
    parser_kennlinie.set_defaults(func=kennlinie)

    parser_control = subparsers.add_parser('c')
    parser_control.set_defaults(func=voltage_control)

    args = parser.parse_args()
    args.func(args)

main()

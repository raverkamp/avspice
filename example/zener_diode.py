import argparse
from avspice import *



def zener(name, v):
    d =Diode("D", 1e-8, 25e-3)
    cutoff = 40
    d =Diode("D", 1e-8, 25e-3, lcut_off=-cutoff, rcut_off=cutoff)
    sc =  SubCircuit(("p", "n"))
    sc.add_component("Df", d, ("p", "n"))
    sc.add_component("Dr", d, ("m", "p"))
    sc.addV("v",v, "n", "m")
    return sc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("v",type=float)

    net = Circuit()
    v = Variable("v")
    net.addV("V", v, "VCC", "0")
    z = zener("Z5", 5)
    net.add_subcircuit("Z", z, ("VCC", "0"))
    ana = Analysis(net)

    args = parser.parse_args()
    print(args)

    res = ana.analyze(variables={"v": args.v})
    res.display()

main()

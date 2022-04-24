""" just another blinker, but this time with an inductor"""
import argparse
import sys
import matplotlib.pyplot as plt
from avspice import Circuit, NPNTransistor, Analysis, pivot

# my transistor model
tr_rav = NPNTransistor("", 1e-12, 25e-3, 100, 10)

def create_circuit(transistor, capacity, inductivity):
    tt = transistor
    net = Circuit()
    net.addV("vc", 10, "v", "0")
    net.addR("rc1",2e3, "v", "t1c")
    net.addR("rc2",2e3, "v", "t2c")
    net.addR("rb1",50e3, "v", "t1b")
    net.addR("rb2",50e3, "v", "t2b")

    t1 = net.add_component("t1",tt, ("t1b", "t1c", "0"))
    t2 = net.add_component("t2",tt, ("t2b", "t2c", "0"))

    net.addR("r1",22e3, "t1c","t2b")

    capa = net.addCapa("capa", capacity, "t2c", "i")
    net.addInduc("ind", inductivity, "i", "t1b")

    return net

def osci(transistor=None,capa=None, ind=None):
    net = create_circuit(transistor, capa, ind)

    ana = Analysis(net)
    res = ana.transient(80e-6,1e-8, capa_voltages={"capa":-0.5}, induc_currents={"ind":-1.8e-4})
    (time,volts,currs) = pivot(res)
    (fig, (p1, p2, p3, p4)) = plt.subplots(4)
    p1.plot(time, currs["t2.E"], label="curr(t2.E)")
    p1.legend()
    p2.plot(time, currs["t1.E"], label="curr(t1.E)")
    p2.legend()
    p3.plot(time, currs["capa.p"], label="curr(capa.p)")
    p3.legend()
    p4.plot(time, volts["capa.p"]-volts["capa.n"], label="volts(capa)")
    p4.legend()
    fig.tight_layout()

    plt.show()

def osci_op(transistor=None,capa=None, ind=None):
    net = create_circuit(transistor, capa, ind)
    ana = Analysis(net)
    res = ana.analyze()
    res.display()

def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    parser_t = subparsers.add_parser('t')
    def transient(args):
        osci(tr_rav, capa= 10e-9, ind=1e-3)
    parser_t.set_defaults(func=transient)

    parser_o = subparsers.add_parser('o')
    def op(args):
        osci_op(tr_rav, capa= 10e-9, ind=1e-3)

    parser_o.set_defaults(func=op)
        
        

    args = parser.parse_args()
    args.func(args)

sys.exit(main())

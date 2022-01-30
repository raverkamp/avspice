""" just another blinker, but this time with an inductor"""
import argparse
import sys
import matplotlib.pyplot as plt
from spice import Network, NPNTransistor, connect, Analysis, pivot

# my transistor model
tr_rav = NPNTransistor(None, "", 1e-12, 25e-3, 100, 10)

def create_circuit(transistor, capacity, inductivity):
    tt = transistor
    net = Network()
    vc = net.addV("vc", 10)
    rc1 = net.addR("rc1",2e3)
    rc2 = net.addR("rc2",2e3)
    rb1 = net.addR("rb1",50e3)
    rb2 = net.addR("rb2",50e3)

    t1 = net.addComp("t1",tt)
    t2 = net.addComp("t2",tt)


    connect(vc.n, net.ground)

    connect(vc.p, rc1.p)
    connect(rc1.n, t1.C)
    connect(t1.E, vc.n)
    connect(vc.p, rb1.p)
    connect(rb1.n, t1.B)

    connect(vc.p, rc2.p)
    connect(rc2.n, t2.C)
    connect(t2.E, vc.n)
    connect(vc.p, rb2.p)
    connect(rb2.n, t2.B)

    rf = net.addR("r1",22e3)
    connect(t1.C, rf.p)
    connect(rf.n, t2.B)

    capa = net.addCapa("capa", capacity)
    ind = net.addInduc("ind", inductivity)

    connect(capa.p, t2.C)
    connect(capa.n, ind.p)
    connect(ind.n, t1.B)

    return net

def osci(transistor=None,capa=None, ind=None):
    net = create_circuit(transistor, capa, ind)

    ana = Analysis(net)
    res = ana.transient(80e-6,1e-8, capa_voltages={"capa":-0.4}, induc_currents={"ind":-1.8e-4})
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

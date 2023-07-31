import matplotlib as mp
import matplotlib.pyplot as plt
from avspice.util import *
import avspice.ncomponents
from avspice import *
import argparse


def fkennlinie(args):
    x = []
    y = []
    fet = ncomponents.NFET(1)
    (fig, ax1) = plt.subplots(1)
    for vgs in [1.01, 2, 3, 4, 5]:
        x = []
        y = []
        for vds in drange(0, 7, 0.01):
            x.append(vds)
            y.append(fet.IS(vgs, vds))
        ax1.plot(x, y, label=f"v(gs)={vgs}")

    ax1.set_xlabel("voltage(ds)")
    ax1.set_ylabel("current(ds)")
    ax1.legend()
    plt.show()


def amplifier(args):
    fet = FET("test", 3)
    net = Circuit()
    net.add_component("fet", fet, ("G", "D", "S"))

    net.addV("vcc", 15, "vc", "0")
    net.addSineV("vc", 1, 10, "input", "0")
    net.addR("RD", 47, "vc", "D")
    net.addR("RS", 15, "S", "0")
    net.addR("Rtop", 200e3, "vc", "G")
    net.addR("Rbot", 200e3, "G", "0")
    net.addCapa("Cin", 0.12e-6, "input", "G")
    net.addCapa("Cout", 0.12e-6, "D", "output")
    net.addR("Rout", 1e3, "output", "0")

    ana = Analysis(net)
    res = ana.transient(1, 0.001, capa_voltages={"Cin": 0, "Cout": 0})

    def currs(x):
        return res.get_current(x)

    def volts(x):
        return res.get_voltage(x)

    time = res.get_time()

    fig, ((a1, a2), (a3, a4)) = plt.subplots(2, 2)

    a1.set_title("Input Voltage")
    a1.plot(time, volts("vc.p"))

    a2.set_title("v(fet.G)")
    a2.plot(time, volts("fet.G"))

    a3.set_title("Output Voltage")
    a3.plot(time, volts("Rout.p"))

    a4.set_title("currs(Rout.p)")
    a4.plot(time, currs("Rout.p"))
    plt.show()


def main():
    parser = argparse.ArgumentParser(prog="Fet Amplifier")
    subparsers = parser.add_subparsers(help="sub-command help", dest="subparser_name")

    parser_f = subparsers.add_parser("k1", help="kennlinie fet")
    parser_f.set_defaults(func=fkennlinie)

    parser_a = subparsers.add_parser("a1", help="fet amplifier")
    parser_a.set_defaults(func=amplifier)

    args = parser.parse_args()

    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_usage()


main()

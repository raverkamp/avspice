"""test and example code for jfet"""
import argparse

import matplotlib.pyplot as plt
from avspice import util, ncomponents, JFET, Network, Circuit, Variable, Analysis


def kennlinie_njfet(args):
    njfet = ncomponents.NJFETn(args.vth, args.beta, vars(args)["lambda"])
    vth = args.vth
    beta = args.beta
    lambda_ = vars(args)["lambda"]

    (fig, ((ax1, ax2, ax2b), (ax3, ax4, ax4b))) = plt.subplots(2, 3)

    fig.suptitle(f"Kennlinie JFET(vth={vth}, beta={beta}, lambda={lambda_})")

    ax1.set_title("current")
    ax1.set_xlabel("voltage(ds)")
    ax1.set_ylabel("current(ds)")

    ax2.set_title("diff current")
    ax2.set_xlabel("voltage(ds)")
    ax2.set_ylabel("d current(ds)/d voltage(ds)")

    ax2b.set_title("diff current, numeric appoximation")
    ax2b.set_xlabel("voltage(ds)")
    ax2b.set_ylabel("current(ds)/d voltage(ds)")

    for fac in [0, 0.2, 0.5, 0.8, 1]:
        vgs = round(fac * args.vth, 3)
        x = []
        y = []
        dy = []
        ndy = []

        for vds in util.drange(0, 7, 0.01):
            x.append(vds)
            y.append(njfet.IS(vgs, vds))
            dy.append(njfet.d_IS_vds(vgs, vds))
            ndy.append(util.ndiff(lambda x: njfet.IS(vgs, x), vds))
        ax1.plot(x, y, label=f"v(gs)={vgs}")
        ax2.plot(x, dy, label=f"v(gs)={vgs}")
        ax2b.plot(x, ndy, label=f"v(gs)={vgs}")

    ax1.legend()
    ax2.legend()
    ax2b.legend()

    ax3.set_title("current")
    ax3.set_xlabel("voltage(gs)")
    ax3.set_ylabel("current(ds)")

    ax4.set_title("diff current")
    ax4.set_xlabel("voltage(ds)")
    ax4.set_ylabel("d current(ds)/d vgs")

    ax4b.set_title("diff current, numeric approximation")
    ax4b.set_xlabel("voltage(ds)")
    ax4b.set_ylabel("d current(ds)/d vgs")

    for fac in [0, 0.2, 0.5, 0.8, 1]:
        vds = round(fac * 7, 2)
        x = []
        y = []
        dy = []
        ndy = []
        for vgs in util.drange(args.vth, 0, 0.01):
            x.append(vgs)
            y.append(njfet.IS(vgs, vds))
            dy.append(njfet.d_IS_vgs(vgs, vds))
            ndy.append(util.ndiff(lambda x: njfet.IS(x, vds), vgs))

        ax3.plot(x, y, label=f"v(ds)={vds}")
        ax4.plot(x, dy, label=f"v(ds)={vds}")
        ax4b.plot(x, ndy, label=f"v(ds)={vds}")

    ax3.legend()
    ax4.legend()
    ax4b.legend()

    fig.tight_layout()
    plt.show()


def kennlinie_njfet_invers(args):
    vth = args.vth
    beta = args.beta
    lambda_ = vars(args)["lambda"]

    njfet = ncomponents.NJFETn(vth, beta, lambda_)
    (fig, ((ax1, ax2, ax2b), (ax3, ax4, ax4b))) = plt.subplots(2, 3)
    fig.suptitle(f"Inverse Kennlinie JFET(vth={vth}, beta={beta}, lambda={lambda_})")

    ax1.set_title("current")
    ax1.set_xlabel("voltage(vds)")
    ax1.set_ylabel("current(ds)")

    ax2.set_title("diff current")
    ax2.set_xlabel("voltage(ds)")
    ax2.set_ylabel("d current(ds)/d ds")

    ax2b.set_title("diff current, numeric appoximation")
    ax2b.set_xlabel("voltage(ds)")
    ax2b.set_ylabel("d current(ds)/d ds")

    ax3.set_title("current")
    ax3.set_xlabel("voltage(gd)")
    ax3.set_ylabel("current(ds)")

    ax4.set_title("diff current")
    ax4.set_xlabel("voltage(ds)")
    ax4.set_ylabel("d current(ds)/d vgs")

    ax4b.set_title("diff current, numeric approximation")
    ax4b.set_xlabel("voltage(ds)")
    ax4b.set_ylabel("d current(ds)/d vgs")

    for fac in [0, 0.2, 0.5, 0.8, 1]:
        vgd = round(fac * args.vth, 3)
        x = []
        y = []
        dy = []
        ndy = []
        for vsd in util.drange(0, 7, 0.01):
            vds = -vsd
            vgs = vds + vgd
            x.append(vds)
            y.append(njfet.IS(vgs, vds))
            dy.append(njfet.d_IS_vds(vgs, vds))
            ndy.append(util.ndiff(lambda x: njfet.IS(vgs, x), vds))

        ax1.plot(x, y, label=f"v(gd)={vgd}")
        ax2.plot(x, dy, label=f"v(gd)={vgd}")
        ax2b.plot(x, ndy, label=f"v(gd)={vgd}")

    ax1.legend()
    ax2.legend()
    ax2b.legend()

    for fac in [0, 0.2, 0.5, 0.8, 1]:
        vds = round(-fac * 7, 3)

        x = []
        y = []
        dy = []
        ndy = []
        for vgd in util.drange(args.vth, 0, 0.01):
            x.append(vgd)
            y.append(njfet.IS(vds, vds + vgd))
            dy.append(njfet.d_IS_vgs(vgs, vds + vgd))
            ndy.append(util.ndiff(lambda x: njfet.IS(x, vgd), vds + vgd))

        ax3.plot(x, y, label=f"v(ds)={vds}")
        ax4.plot(x, dy, label=f"v(ds)={vds}")
        ax4b.plot(x, dy, label=f"v(ds)={vds}")

    ax3.legend()
    ax4.legend()
    ax4b.legend()

    fig.tight_layout()
    plt.show()


def simple(args):
    from avspice.circuits import JFET

    vth = -2
    beta = 1  # args.beta
    lambda_ = vars(args)["lambda"]

    jfet = JFET("jfet", vth, beta, lambda_)

    net = Circuit()
    net.add_component("jfet", jfet, ("G", "D", "0"))

    vcc = 9

    net.addV("vcc", vcc, "D", "0")

    net.addV("vc", Variable("vc"), "G", "0")

    ana = Analysis(net)
    vc = []
    current_d = []
    for v in util.drange(vth, 0, 0.01):
        res = ana.analyze(variables={"vc": v})
        vc.append(v)
        current_d.append(-res.get_current("jfet.D"))

    (fig, ax1) = plt.subplots(1)
    fig.suptitle(f"Inverse Kennlinie JFET(vth={vth}, beta={beta}, lambda={lambda_})")
    ax1.plot(vc, current_d)
    fig.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(prog="JFet")
    subparsers = parser.add_subparsers(help="sub-command help", dest="subparser_name")

    parser_k = subparsers.add_parser("k", help="kennlinie N-Jfet")
    parser_k.set_defaults(func=kennlinie_njfet)

    parser_k.add_argument("vth", type=float)
    parser_k.add_argument("beta", type=float)
    parser_k.add_argument("lambda", type=float)

    parser_ki = subparsers.add_parser("ki", help="kennlinie N-Jfet, invers")
    parser_ki.set_defaults(func=kennlinie_njfet_invers)

    parser_ki.add_argument("vth", type=float)
    parser_ki.add_argument("beta", type=float)
    parser_ki.add_argument("lambda", type=float)

    parser_simple = subparsers.add_parser("simple", help="simple circuit")
    parser_simple.add_argument("lambda", type=float)

    parser_simple.set_defaults(func=simple)

    args = parser.parse_args()

    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_usage()


main()

from avspice import *


# transistor_gain is the gain of the tarnssistor
# r_ca is resitor value in front of capacitor
def create_blinker(transistor_gain=None, cutoff=40, rampup=None):
    # https://www.elektronik-labor.de/Lernpakete/Blinker.html
    # weiter unten
    # oder hier: https://www.elektronik-labor.de/Lernpakete/Kalender08/Kalender08.htm#_Toc197001462
    transistor_gain = 100 if transistor_gain is None else transistor_gain
    tt = NPNTransistor("", 1e-12, 25e-3, transistor_gain, 10, cutoff=cutoff)
    net = Circuit()
    d = Diode("D", 1e-8, 25e-3, lcut_off=-cutoff, rcut_off=cutoff)
    if rampup is None or rampup <= 0:
        net.addV("vc", Variable("vc"), "v", "0")
    else:
        v = PieceWiseLinearVoltage("x", [(0, 0), (rampup, Variable("vc"))])
        net.add_component("vc", v, ("v", "0"))
    net.add_component("d1", d, ("v", "da"))
    net.addR("r1", 27e3, "t1c", "t1b")
    net.addR("r2", 27e3, "v", "t1c")
    net.addR("r3", 1e3, "da", "t2c")

    ca = net.addCapa("ca", Variable("capa", 10e-6), "t2c", "t1b")
    #  ca = net.addR("ca", 10e6)
    net.add_component("t1", tt, ("t1b", "t1c", "0"))
    net.add_component("t2", tt, ("t1c", "t2c", "0"))
    return net


def create_current_mirror_npn(model):
    assert isinstance(model, NPNTransistor)
    sc = SubCircuit(("inc", "outc", "00"))
    sc.add("Tin", model, ("inc", "inc", "00"))
    sc.add("Tout", model, ("inc", "oiutc", "00"))
    return sc


def create_current_mirror_pnp(model):
    assert isinstance(model, PNPTransistor)
    sc = SubCircuit(("inc", "outc", "top"))
    sc.add("Tin", model, ("inc", "inc", "top"))
    sc.add("Tout", model, ("inc", "outc", "top"))
    return sc


def create_current_source_npn(
    tmodel: NPNTransistor, zdiode_voltage: float, current: float
):
    assert isinstance(tmodel, NPNTransistor)

    r = (zdiode_voltage - 0.3) / current
    zdiode_model = ZDiode("Z0", zdiode_voltage, 1e-8, 25e-3)

    sc = SubCircuit(("vc", "top", "bot"))
    sc.addR("RB", 1e3, "vc", "B")
    sc.add("Q", tmodel, ["B", "top", "Rout"])
    sc.addR("RBot", r, "Rout", "bot")
    sc.add("ZD", zdiode_model, ["bot", "B"])
    return sc

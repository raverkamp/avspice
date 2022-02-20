from spice import *

# transistor_gain is the gain of the tarnssistor
# r_ca is resitor value in front of capacitor
def create_blinker(transistor_gain=None):
    # https://www.elektronik-labor.de/Lernpakete/Blinker.html
    # weiter unten
    # oder hier: https://www.elektronik-labor.de/Lernpakete/Kalender08/Kalender08.htm#_Toc197001462
    transistor_gain = 100 if transistor_gain is None else transistor_gain 
    tt = NPNTransistor("", 1e-12, 25e-3, transistor_gain, 10) 
    net = Network()
    net.addV("vc",  Variable("vc"), "v", "0")
    net.addD("d1", 1e-8, 25e-3, "v", "da")
    net.addR("r1", 27e3, "t1c", "t1b")
    net.addR("r2", 27e3,"v", "t1c")
    net.addR("r3", 1e3, "da", "t2c")

    ca = net.addCapa("ca", Variable("capa", 10e-6), "t2c", "t1b")
  #  ca = net.addR("ca", 10e6)
    net.add_component("t1",tt, ("t1b", "t1c", "0"))
    net.add_component("t2",tt, ("t1c", "t2c", "0"))
    return net
    

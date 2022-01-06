from spice import *

# transistor_gain is the gain of the tarnssistor
# r_ca is resitor value in front of capacitor
def create_blinker(transistor_gain=None, r_ca=1):
    # https://www.elektronik-labor.de/Lernpakete/Blinker.html
    # weiter unten
    # oder hier: https://www.elektronik-labor.de/Lernpakete/Kalender08/Kalender08.htm#_Toc197001462
    transistor_gain = 100 if transistor_gain is None else transistor_gain 
    tt = NPNTransistor(None, "", 1e-12, 25e-3, transistor_gain, 10) 
    net = Network()
    vc = net.addV("vc",  Variable("vc")) #?
    d1 = net.addD("d1", 1e-8, 25e-3)
    r1 = net.addR("r1", 27e3)
    r2 = net.addR("r2", 27e3)
    r3 = net.addR("r3", 1e3)
    r_ca = net.addR("r_ca",r_ca)

    ca = net.addCapa("ca", 10e-6)
  #  ca = net.addR("ca", 10e6)
    t1 = net.addComp("t1",tt)
    t2 = net.addComp("t2",tt)
    connect(vc.p, r2.p)
    connect(vc.p, d1.p)

    connect(r2.n, t1.C)
    connect(r2.n, r1.p)
    connect(r2.n, t2.B)
    
    connect(r1.n, ca.n)
    connect(r1.n, t1.B)

    connect(d1.n, r3.p)
    connect(ca.p, r_ca.n) 
    connect(r_ca.p, r3.n)
    
    connect(r3.n, t2.C)

    connect(t1.E, net.ground)
    connect(t2.E, net.ground)

    connect(vc.n, net.ground)
    return net
    

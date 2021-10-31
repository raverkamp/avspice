import matplotlib as mp
import matplotlib.pyplot as plt
from spice import *
import argparse
import sys
from util import *


def create_blinker():
    # https://www.elektronik-labor.de/Lernpakete/Blinker.html
    # weiter unten
    # oder hier: https://www.elektronik-labor.de/Lernpakete/Kalender08/Kalender08.htm#_Toc197001462
    tt = NPNTransistor(None, "", 1e-12, 25e-3, 100, 10) 
    net = Network()
    vc = net.addV("vc",  Variable("vc")) #?
    d1 = net.addD("d1", 1e-8, 25e-3)
    r1 = net.addR("r1", 27e3)
    r2 = net.addR("r2", 27e3)
    r3 = net.addR("r3", 1e3)

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
    connect(r3.n, ca.p)
    connect(r3.n, t2.C)

    connect(t1.E, net.ground)
    connect(t2.E, net.ground)

    connect(vc.n, net.ground)
    return net
    

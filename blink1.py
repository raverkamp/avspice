import matplotlib as mp
import matplotlib.pyplot as plt
from spice import *
import argparse
import sys
from util import *


def create_blinker():
    # https://www.elektronik-labor.de/Lernpakete/Blinker.html
    tt = NPNTransistor(None, "", 1e-12, 25e-3, 100, 10) 
    net = Network()
    vc = net.addV("vc",  Variable("vc")) #?
    d1 = net.addD("d1", 1e-8, 25e-3)
    r1 = net.addR("r1", 27e3)
    r2 = net.addR("r2", 27e3)
    r3 = net.addR("r3", 1e3)

    rt2e = net.addR("rt2e", Variable("rt2e"))
    rt1e = net.addR("rt1e", Variable("rt1e"))
    ca = net.addCapa("ca", 10e-6)
  #  ca = net.addR("ca", 1e-1)
    t1 = net.addComp("t1",tt)
    t2 = net.addComp("t2",tt)
    connect(vc.p, r2.p)
    connect(vc.p, d1.p)

    connect(r2.n, t1.C)
    connect(r2.n, r1.p)
    connect(r2.n, t2.B)
    
    connect(r1.n, ca.p)
    connect(r1.n, t1.B)

    connect(d1.n, r3.p)
    connect(r3.n, ca.n)
    connect(r3.n, t2.C)

    connect(t1.E, rt1e.p)
    connect(rt1e.n, vc.n)
    connect(t2.E, rt2e.p)
    connect(rt2e.n,vc.n)

    connect(vc.n, net.ground)
    return net
    
def blinker(args):
    net = create_blinker()
    ana = Analysis(net)

    base_vca = 0 #-5 # -8.25

    rt2e= 1
    rt1e = 1

    sol = None
    for x in [0.001, 0.01, 0.1,0.2,0.25,0.27,0.28, 0.29,0.295,0.299, 0.2995, 0.29995 ,0.3001,0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1,2,4,9]: #,20,30,40,45, 47,47.5,47.7,478.9,48, 50,100, 200,300] : # 100, 900, 1000]:
        res = ana.analyze(maxit=50, start_solution_vec=sol, capa_voltages={"ca": base_vca},
                          variables={"vc": x, "rt2e": rt2e, "rt1e": rt1e})
        print("a", x,res)
        sol = res.solution_vec
    print("---------------------------------------------------")
    ch = 0

    xs = []
    ca_v = []
  
    t1b_v = []
    t2c = []
    t1c = []
    t1b = []
    it1b = []
    t2e = []
    it1c = []
    it2c = []
    it2b = []
    itd  = []
    iters = []
    s = 0.0001/10
    x = 0
    v_ca = base_vca
    switched = False
    while x < 3:
        ok = False
        maxit = 50
        res = ana.analyze(maxit=maxit, start_solution_vec=sol, capa_voltages={"ca": v_ca },
                              variables={"vc": 9, "rt2e": rt2e, "rt1e": rt1e})
        if isinstance(res,str):
            print(x, res)
            break
        ca_cu = res.get_current("ca.p")

        capa = net.get_object("ca").capa
        vca_new = v_ca + s*ca_cu/capa
        print(("OK", x, ca_cu,v_ca, vca_new))
        v_ca = vca_new
        xs.append(x)

        ca_v.append(v_ca)
        t1b_v.append(res.get_voltage("t1.B"))
        t2c.append(res.get_voltage("t2.C"))
        t1c.append(res.get_voltage("t1.C"))
        t1b.append(res.get_voltage("t1.B"))
        t2e.append(res.get_voltage("t2.E"))
        it2c.append(res.get_current("t2.C"))
        it2b.append(res.get_current("t2.B"))
        it1b.append(res.get_current("t1.B"))
        iters.append(res.iterations)
        itd.append(res.get_current("d1.p"))
        #   print((x, res.iterations, ch, ica))
        sol = res.solution_vec

        x += s
    fig, ((a1,a2, a3), (a4, a5, a6)) = plt.subplots(2,3)

    a1.set_title("current d1, current t1b")
    a1.plot(xs, itd)
    a1.plot(xs, it1b)
    
    a2.set_title("capa voltage=blue and t1.B voltage=red")
    a2.plot(xs, ca_v, label="capa voltage")
    a2.plot(xs, t1b_v, label="t1b voltage")
    a2.legend()
    
    a3.plot(xs, t2c, label="t2c")
    a3.plot(xs, t1c, label="t1c")
    a3.plot(xs, t1b, label="t1b")
    a3.plot(xs, t2e, label="t2e")
    a3.set_title("Voltages")
    a3.legend()
    a4.plot(xs, it2c)
    a4.set_title("it2c")

    a5.plot(xs, it2b)
    a5.set_title("it2b")

    a6.set_title("iterations")
    a6.plot(xs, iters)
    
    plt.ion()
    plt.show()
    input()
    
def main():
    (cmd, args) = getargs()

    if cmd == "b":
        blinker(args)
    else:
        raise Exception("unknown command: {0}".format(cmd))
    
main()

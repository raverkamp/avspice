import matplotlib as mp
import matplotlib.pyplot as plt
from spice import *
import argparse
import sys
from util import *

import math




def plot1(args):
    # das passt alles nicht
    # oder doch: Basis Spannug sollte über der vom Kollektor liegen
    t = NPNTransistor(None, "", 1e-12, 25e-3, 100, 10)
    vbl = [0, 0.1, 0.2, 0.3,0.4]
    k = math.ceil(math.sqrt(len(vbl)))
    (f,plts) = plt.subplots(k,k)
    ve = 0
    x = list(drange(0, 0.3, 0.01))
    i = 0
    for vb in vbl:
        ax = plts[i//k][i%k]
        ie = [t.IE(vb-ve, vb-vc) for vc in x]
        ib = [t.IB(vb-ve, vb-vc) for vc in x]
        te = "npn transistor, vb={0}, ve={1}".format(vb, ve)
        ax.set_title(te)
        ax.set_ylabel("current")
        ax.set_xlabel("vc")
        ax.plot(x,ie, label="ie")
        ax.plot(x,ib, label="ib")
        ax.legend()
        i+=1

    plt.ion()
    plt.show()
    input()

def plot2(args):
    t = NPNTransistor(None, "", 1e-12, 25e-3, 100, 10)

    vc = 2
    ve = 0
    x = list(drange(-0.5,3.5, 0.01)    )
    ie = [t.IE(vb-ve, vb-vc) for vb in x]
    ib = [t.IB(vb-ve, vb-vc) for vb in x]
    fig, ax = plt.subplots()  # Create a figure containing a single axes.
    plt.plot(x,ie, color="black")
    plt.plot(x,ib, color="green")
    plt.show()
    input()

def plot3(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('cpot', type=float)
    parser.add_argument('cutoff', type=float)

    args = parser.parse_args(args)
    t = NPNTransistor(None, "", 1e-12, 25e-3, 100, 10)
    x = list(drange(-0.5, 3.5, 0.01))
    vc = args.cpot
    ve = 0
    ie = [t.IE(vb-ve, vb-vc) for vb in x]
    ib = [t.IB(vb-ve, vb-vc) for vb in x]
    ic = [t.IC(vb-ve, vb-vc) for vb in x]
    plt.plot(x,ie, color="black")
    plt.plot(x,ib, color="green")
    plt.plot(x,ic, color="blue")
    plt.show()
    input()


def plot4(args):
    x = list(drange(-2, 10, 0.01))
    y = []
    z = []
    sol = None
    iy = []

    tt = NPNTransistor(None, "", 1e-12, 25e-3, 100, 10)
 
    net = Network()
    vc = net.addV("vc", 2)
    vb = net.addV("vb", Variable("vb"))
    re = net.addR("re", 100)
    rb = net.addR("rb", 10e3)
    t1 = net.addComp("T1", tt)
    connect(vc.p, t1.C)
    connect(vc.n, net.ground)
    connect(vb.p, rb.p)
    connect(rb.n, t1.B)
    connect(vb.n, net.ground)
    connect(t1.E, re.p)
    connect(re.n, net.ground)
    ana = Analysis(net)
    for v in x:
        res = ana.analyze(maxit=30, start_solution_vec=sol, variables={"vb": v})
        if isinstance(res, str):
            print("no covergence at: {0}".format(v))
            y.append(None)
            z.append(None)
            iy.appned(None)
            sol = None
        else:
            y.append(res.get_current(t1.E))
            z.append(res.get_current(t1.B))
            iy.append(res.iterations)
            sol = res.solution_vec
    fig, (ax1,ax) = plt.subplots(2)
    ax1.plot(x,y, color="black", label="I(E)")
    ax2 = ax1.twinx()
    ax2.plot(x,z, color="green", label="I(B)")
    ax1.legend()
    ax2.legend()
    
    fig.tight_layout()
    ax.plot(x, iy)
    ax.set_title("Iterations")
    plt.show()
    #input()

def plot5(args):
    x = list(drange(-2, 10, 0.01))
    y = []
    z = []
    sol = None
    iy = []

    tt = NPNTransistor(None, "", 1e-12, 25e-3, 100, 10)
    net = Network()
    vc = net.addV("vc", 5)
    rc = net.addR("rc", 1e9)
    rb = net.addR("rb", Variable("rb"))
    t1 = net.addComp("T1", tt)
    connect(vc.p, rc.p)
    connect(rc.n, t1.C)
    connect(vc.p, rb.p)
    connect(rb.n, t1.B)
    connect(vc.n, net.ground)
    connect(t1.E, net.ground)

    ana = Analysis(net)
    sol = None
    lrb = []
    lvb = []
    for vrb in drange(1e3,1e6,1000):
        res = ana.analyze(maxit=30, start_solution_vec=sol, variables={"rb": vrb})
        if isinstance(res, str):
            print("no covergence at: {0}".format(v))
            y.append(None)
            z.append(None)
            iy.append(None)
            sol = None
        else:
            lrb.append(vrb)
            lvb.append(res.get_voltage (t1.B))
            sol = res.solution_vec
    fig, (ax1,ax2) = plt.subplots(2)
    ax1.plot(lrb,lvb,color="black")
    fig.tight_layout()
    plt.show()
    #input()


def emitter(args):
    tt = NPNTransistor(None, "", 1e-12, 25e-3, 100, 10)
    nw = Network()
    net = Network()
    vc = net.addV("vc", 5)
    r1 = net.addR("r1", 2000)
    r2 = net.addR("r2", 2000)


    vb = net.addV("vb", Variable("vb"))
    rc = net.addR("rc", 1e3)
    rb = net.addR("rb", 1e5)

    t1 = net.addComp("T1", tt)

    connect(vc.p, rc.p)
    connect(vc.n, net.ground)
    connect(rc.n, t1.C)
    connect(t1.E, net.ground)
    connect(r1.p, vc.p)
    connect(r1.n, r2.p)
    connect(r2.n, net.ground)
    connect(r1.n, t1.B)

    connect(vb.p, rb.p)
    connect(vb.n, net.ground)
    connect(rb.n, t1.B)


    y = []
    z = []
    ana = Analysis(net)
    sol = None
    x = list(drange(0,0.1,0.001))
    for v in x:
        res = ana.analyze(maxit=30, start_solution_vec=sol, variables={"vb": v})
        if isinstance(res, str):
            print("no covergence at: {0}".format(v))
            y.append(None)
            sol = None
        else:
            y.append(res.get_voltage(t1.C))
            z.append(res.get_voltage(t1.B))
            sol = res.solution_vec
    fig, (a1,a2) = plt.subplots(2)
    a1.plot(x,y, color="blue")
    a2.plot(x,z, color="blue")
    print(z)
    plt.show()

def capa(args):
    net = Network()
    vc = net.addV("vc", 2)
    r = net.addR("r", 1e4)
    c = net.addCapa("ca", 100e-6)
    connect(vc.p, c.p)
    connect(c.n, r.p)
    connect(r.n, vc.n)
    connect(vc.n, net.ground)
    ana = Analysis(net)
    ch = 0

    xs = []
    ys = []
    vcp = []
    vcn = []
    s = 0.2
    x = 0
    while x < 10:
        res = ana.analyze(maxit=30, charges={"ca": ch })
        ica = res.get_current(c.p)
        ch += s * ica
        x += s
        xs.append(x)
        ys.append(ch)
        vcp.append(res.get_voltage("ca.p"))
        vcn.append(res.get_voltage("ca.n"))
    fig, (a1,a2) = plt.subplots(2)
    a1.set_title("charge")
    a1.plot(xs,ys, color="blue")
    a2.set_title("voltage1 ,vcp blue, vcn red")
    a2.plot(xs, vcp, color="blue")
    a2.plot(xs, vcn, color="red")
    plt.show()

def create_blinker():
    # https://www.elektronik-labor.de/Lernpakete/Blinker.html
    tt = NPNTransistor(None, "", 1e-12, 25e-3, 100, 10)
    net = Network()
    vc = net.addV("vc",  Variable("vc", 7)) #?
    d1 = net.addD("d1", 1e-8, 25e-3)
    rd= net.addR("rd", Variable("rd",1e3)) # resistor parallel to diode
    r1 = net.addR("r1", 2.7e3)
    r2 = net.addR("r2", 27e3)
    r3 = net.addR("r3", 1e3)

    rt2e = net.addR("rt2e", Variable("rt2e",1))
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

    connect(t1.E, vc.n)
    connect(t2.E, rt2e.p)
    connect(rt2e.n,vc.n)

    connect(vc.n, net.ground)

    connect(d1.p, rd.p)
    connect(d1.n, rd.n)
    
    return net

def create_blinker2():
    """https://www.elektronik-kompendium.de/sites/praxis/bausatz_led-wechselblinker.htm"""
    tt = NPNTransistor(None, "", 1e-12, 25e-3, 100, 10)
    net = Network()
    vc = net.addV("vc",  Variable("vc",7)) #?
    p1 = net.addR("p1", 50e3)
    d1 = net.addD("d1", 1e-8, 25e-3)
    d2 = net.addD("d2", 1e-8, 25e-3)
    r1 = net.addR("r1", 470 + 100)
    r2 = net.addR("r2", 470)
    r3 = net.addR("r3", 3.9e3)
    r4 = net.addR("r4", 3.9e3)
    c1 = net.addCapa("c1", 47e-6)
    c2 = net.addCapa("c2", 47e-6)
    t1 = net.addComp("t1", tt)
    t2 = net.addComp("t2", tt)

   
    cg1 = net.addCapa("cg1", 1e-6)
    cg2 = net.addCapa("cg2", 1.5e-6)

    connect(cg1.p, t1.B)
    connect(cg2.p, t2.B)
    connect(cg1.n, net.ground)
    connect(cg2.n, net.ground)
   
    

    
    
    
    connect(vc.n, net.ground)
    connect(vc.p, p1.p)
    connect(vc.p, d1.p)
    connect(vc.p, d2.p)
    connect(p1.n, r3.p)
    connect(p1.n, r4.p)
    connect(d1.n, r1.p)
    connect(d2.n, r2.p)
    connect(r1.n, c1.p)
    connect(r2.n, c2.p)
    connect(r3.n, c1.n)
    connect(r4.n, c2.n)
    connect(c1.n, t2.B)
    connect(c2.n, t1.B)
    connect(r1.n, t1.C)
    connect(r2.n, t2.C)
    connect(t1.E, net.ground)
    connect(t2.E, net.ground)
    
    return net
    
    
def blinker(args):
    net = create_blinker()
    ana = Analysis(net)
    voltage = 7

    base_vca = 0

    sol = None
    
    res = ana.analyze(maxit=50, start_solution_vec=sol,
                      capa_voltages={"ca": base_vca},
                      variables={"vc": voltage, "rt2e": 0.001})
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
    s = 0.00001
    x = 0
    v_ca = base_vca
  
    while x < 1:
        ok = False
        try:
            res = ana.analyze(maxit=100, start_solution_vec=sol,
                              capa_voltages={"ca": v_ca },
                              variables={"vc": voltage, "rt2e": 0.001})
        except e:
            print(e)
            break
        if isinstance(res,str):
            print(x, res)
            if sol is None:
                break
            # neu start der Suche, aber nur einmal
            sol = None
            continue
        ca_cu = res.get_current("ca.p")

        capa = net.get_object("ca").capa
        vca_new = v_ca + s*ca_cu/capa
        print(("OK", x, ca_cu,v_ca, vca_new, res.mat_cond))
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
    a2.plot(xs, ca_v, color="blue")
    a2.plot(xs, t1b_v, color ="red")

    a3.plot(xs, t2c, color ="blue")
    a3.plot(xs, t1c, color ="green")
    a3.plot(xs, t1b, color ="red")
    a3.plot(xs, t2e, color ="pink")
    a3.legend()
    a3.set_title("T2 C voltage blue T1 C voltage green, t1b red, t1e pink")

    a4.plot(xs, it2c)
    a4.set_title("it2c")

    a5.plot(xs, it2b, label="I(tb2)")
    a5.plot(xs, it1b, label="I(tb1)")
    a5.set_title("Base currents")
    a5.legend()

    a6.set_title("iterations")
    a6.plot(xs, iters)


    plt.show()

def transient(ana, variables, start_sol, limit, step):
    capas = ana.netw.get_capacitors()
    capa_voltages = {}
    for c in capas:
        capa_voltages[c.name] = 0
    t = 0
    ts = []
    vals = {}
    sol = start_sol
    smaller = False
    while t <limit:
        print(t)
        ok = False
        res = ana.analyze(maxit=100, start_solution_vec=sol,
                          capa_voltages=capa_voltages,
                          variables=variables, alpha=1)
        if isinstance(res,str):
            print(t, res)
            print(capa_voltages)
            break
        capa_voltages_new = {}
        dv = 0
        for c in capas:
            dv = max(dv, abs(res.get_current(c.p)) / c.capa)
        dv = max(0.01,dv)
        real_step = min(step, 0.01/dv)
        for c in capas:
            ca_cu = res.get_current(c.p)
            capa_voltages_new[c.name] =  capa_voltages[c.name] + real_step*ca_cu/c.capa
        capa_voltages = capa_voltages_new
        ts.append(t)
        for port in res.voltages:
            v_port = res.get_voltage(port)
            x = "v." + port.pname()
            if not x in vals:
                vals[x] = []
            vals[x].append(v_port)
        for port in res.currents:
            c_port = res.get_current(port)
            x = "c." + port.pname()
            if not x in vals:
                vals[x] = []
            vals[x].append(c_port)
        
        
        t+=real_step
    return (ts, vals)
    
    
    

def blinker2(args):
    net = create_blinker2()
    ana = Analysis(net)
    sol = None
    for x in [0.001, 0.01, 0.02,0.05,0.06,0.07, 0.08,  0.1,0.15, 0.2,0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
        #,20,30,40,45, 47,47.5,47.7,478.9,48, 50,100, 200,300] : # 100, 900, 1000]:
        res = ana.analyze(maxit=50, start_solution_vec=sol,
                          capa_voltages={},
                          variables={}, energy_factor=x)
        #print(x,res)
        sol = res.solution_vec
    print("---------------------------------------------------")
    (t, x) = transient(ana, {}, sol, 20, 0.001)
    (f,(p1, p2, p3)) = plt.subplots(3)
    #p1.set_titile("volts!")
    p1.plot(t, x["v.r1.p"], label="v.r1.p")
    p1.plot(t, x["v.r2.p"], label="v.r2.p")
    plt.show()

def blinker2b(args):
    net = create_blinker2()
    ana = Analysis(net)
    sol = None
    capa_voltages = {'c1': 5.626925210387078, 'c2': -0.4547942615309388}
    for x in [0.001, 0.01, 0.02,0.05,0.06,0.07, 0.08,  0.1,0.15, 0.2,0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
        #,20,30,40,45, 47,47.5,47.7,478.9,48, 50,100, 200,300] : # 100, 900, 1000]:
        res = ana.analyze(maxit=50, start_solution_vec=sol,
                          capa_voltages=capa_voltages,
                          variables={}, energy_factor=x)
        #print(x,res)
        sol = res.solution_vec
    print(res.solution_vec)

def blinker3(args):
    net = create_blinker()
    ana = Analysis(net)
    sol = None
    for x in [0.001, 0.01, 0.02,0.05,0.06,0.07, 0.08,  0.1,0.15, 0.2,0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
        #,20,30,40,45, 47,47.5,47.7,478.9,48, 50,100, 200,300] : # 100, 900, 1000]:
        res = ana.analyze(maxit=50, start_solution_vec=sol,
                          capa_voltages={"cax": -1},
                          variables={}, energy_factor=x)
        #print(x,res)
        sol = res.solution_vec
    print("---------------------------------------------------")
    (t, x) = transient(ana, {}, sol, 1, 0.001)
    (f,(p1, p2, p3)) = plt.subplots(3)
    #p1.set_titile("volts!")
    p1.plot(t, x["v.ca.p"], label="v.ca.p")
    p1.plot(t, x["v.ca.n"], label="v.ca.n")
    plt.show()
    
def main():
    (cmd, args) = getargs()
    if cmd == "1":
        plot1(args)
    elif cmd == "2":
        plot2(args)
    elif cmd == "3":
        plot3(args)
    elif cmd == "4":
        plot4(args)
    elif cmd == "5":
        plot5(args)
    elif cmd == "e":
        emitter(args)
    elif cmd == "c":
        capa(args)
    elif cmd == "b":
        blinker(args)
    elif cmd == "b2":
        blinker2(args)
    elif cmd == "b2b":
        blinker2b(args)
    elif cmd == "b3":
        blinker3(args)
    else:
        raise Exception("unknown commnd: {0}".format(cmd))

main()

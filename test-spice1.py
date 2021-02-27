3import spice
from spice import Network, connect, compute_nodes, Analysis, Diode, NPNTransistor
import argparse
import pprint as pp



def simple_current():
    net = Network()
    r1 = net.addR("r1",100)
    c1 = net.addC("c1", 1)
    connect(c1.p, r1.p)
    connect(r1.n, c1.n)
    connect(r1.n, net.ground)
    return net

def simple_current2():
    net = Network()
    r1 = net.addR("r1",100)
    r2 = net.addR("r2",100)
    r3 = net.addR("r3",100)
    c1 = net.addC("c1", 1)
    connect(c1.p, r1.p)
    connect(r1.n, c1.n)
    connect(r1.n, net.ground)
    connect(r1.n, r2.n)
    connect(r1.p, r2.p)
    connect(r1.n, r3.n)
    connect(r1.p, r3.p)
    analy = Analysis(net)
    r = analy.analyze()
    pp.pprint(r)

def test1():
    net = simple_current()
    analy = Analysis(net)
    r = analy.analyze()
    pp.pprint(analy.solution_vec)
    pp.pprint(r)

def test2c():

    net = Network()
    r1 = net.addR("r1",100)
    c1 = net.addC("c1", 1)
    r2 = net.addR("r2", 200)
    connect(c1.p, r1.p)
    connect(r1.n, c1.n)
    connect(r2.p, c1.p)
    connect(r2.n, c1.n)
    connect(c1.n, net.ground)
    ana = Analysis(net)
    pp.pprint(ana.analyze())

def test2v():
    net = Network()
    r1 = net.addR("r1",100)
    v1 = net.addV("c1", 1)
    r2 = net.addR("r2", 200)

    connect(v1.n, net.ground)

    connect(v1.p, r1.p)
    connect(r1.n, v1.n)
    connect(r2.p, v1.p)
    connect(r2.n, v1.n)
    connect(v1.n, net.ground)
    ana = Analysis(net)
    pp.pprint(ana.analyze())

def test2b():
    net = Network()
    r1 = net.addR("r1",100)
    c1 = net.addC("c1", 1)
    r2 = net.addR("r2", 200)
    connect(c1.n, r1.p)
    connect(r1.n, c1.p)
    connect(r2.p, c1.p)
    connect(r2.n, c1.n)
    n1 = net.addN("N1")
    connect(n1, c1.p)
    n2 = net.addN("N2")
    connect(n2, c1.n)
    connect(n1, net.ground)
    ana = Analysis(net)
    pp.pprint(ana.analyze())

def test3():
    net = Network()
    c1 = net.addC("c1", 1)
    n1 = net.addN("n1")
    connect(c1.n, net.ground)
    connect(c1.p, n1)
    r1 = net.addR("r1",10)
    r2 = net.addR("r2",20)
    r3 = net.addR("r3", 5)
    connect(c1.p, r1.p)
    connect(r1.n, r2.p)
    connect(r2.n, r3.p)
    connect(r3.n, net.ground)

    ana = Analysis(net)
    pp.pprint(ana.analyze())
    xx = spice.sym_analyze(net)

def test4():
    net = Network()
    v1 = net.addV("v1", 1)
    v2 = net.addV("v2", 2)
    r1 = net.addR("r1",10)
    n1 = net.addN("n1")
    conenct(n1, v1.p)
    connect(v1.n, v2.p)
    connect(v2.n, r1.p)
    connect(v1.p, net.ground)
    connect(r1.n, net.ground)
    ana = Analysis(net)
    pp.pprint(ana.analyze())

def test5():
    net = Network()
    v1 = net.addV("v1", 10)
    r1 = net.addR("r1", 100)
    d1 = net.addD("d1", 1e-8, 25e-3)
    n1 = net.addN("n1")
    connect(n1, v1.p)
    connect(v1.n, net.ground)
    connect(v1.p, r1.p)
    connect(r1.n, d1.p)
    connect(d1.n, net.ground)
    ana = Analysis(net)
    pp.pprint(ana.analyze())

def test5a():
    net = Network()
    v1 = net.addV("v1", 10)
    r1 = net.addR("r1", 100)
    d1 = net.addD("d1", 1e-8, 25e-3)
    d2 = net.addD("d2", 1e-8, 25e-3)
    n1 = net.addN("n1")
    connect(n1, v1.p)
    connect(v1.n, net.ground)
    connect(v1.p, r1.p)
    connect(r1.n, d1.p)
    connect(d1.n, d2.p)
    connect(d2.n, v1.n)
    ana = Analysis(net)
    pp.pprint(ana.analyze())

def test5b():
    net = Network()
    c1 = net.addC("c1", 0.1)
    r1 = net.addR("r1", 100)
    d1 = net.addD("d1", 1e-8, 25e-3)
    d2 = net.addD("d2", 1e-8, 25e-3)
    n1 = net.addN("n1")
    connect(n1, c1.p)
    connect(c1.n, net.ground)
    connect(c1.p, r1.p)
    connect(r1.n, d1.p)
    connect(d1.n, d2.p)
    connect(d2.n, c1.n)
    ana = Analysis(net)
    pp.pprint(ana.analyze())


def test6():
    net = Network()
    n1 = net.addN("n1")
    c1 = net.addC("c1", 5)
    r1 = net.addR("r1", 100)
    d1 = net.addD("d1", 1e-8, 25e-3)
    connect(n1, c1.p)
    connect(c1.n, net.ground)
    connect(c1.p, d1.p)
    connect(c1.n, d1.n)
    connect(c1.p,r1.p)
    connect(c1.n,r1.n)
    ana = Analysis(net)
    pp.pprint(ana.analyze())

def test7():
    net = Network()
    v1 = net.addV("v1", 10)
    connect(v1.n, net.ground)
    port = v1.p
    for i in range(10):
        r = net.addR("r" +repr(i),10)
        connect(r.p, port)
        d = net.addD("d"+ repr(i), 1e-8, 25e-3)
        connect(r.n, d.p)
        port = d.n
    connect(port, net.ground)
    ana = Analysis(net)
    pp.pprint(ana.analyze())


def test8a():
    net = Network()
    dt = Diode(None, "", 1e-8, 25e-3, 10)
    v1 = net.addV("v1", 2)
    r1 = net.addR("r1", 100)
    d1 = net.addComp("d1",dt)
    n1 = net.addN("n1")
    connect(v1.n, net.ground)
    connect(v1.p, r1.p)
    connect(r1.n, d1.p)
    connect(n1, d1.p)
    connect(d1.n, net.ground)
    ana = Analysis(net)
    pp.pprint(ana.analyze())

def test8b():
    net = Network()
    dt = Diode(None, "", 1e-8, 25e-3, 10)
    c1 = net.addC("c1", 2)
    r1 = net.addR("r1", 100)
    d1 = net.addComp("d1",dt)
    n1 = net.addN("n1")
    connect(c1.n, net.ground)
    connect(c1.p, d1.p)
    connect(d1.n, r1.p)
    connect(n1, d1.p)
    connect(r1.n, net.ground)
    ana = Analysis(net)
    pp.pprint(ana.analyze())

def test8c():
    net = Network()
    dt = Diode(None, "", 1e-8, 25e-3, 10)
    c1 = net.addC("c1", 1)
    d1 = net.addComp("d1",dt)
    n1 = net.addN("n1")
    connect(c1.n, net.ground)
    connect(c1.p, d1.p)
    connect(d1.n, c1.n)
    connect(n1, c1.p)
    ana = Analysis(net)
    pp.pprint(ana.analyze())



def test9():
    net = Network()
    v1 = net.addV("v1", 2)
    vb = net.addV("vb",0.5)
    re = net.addR("re", 100)
    rb = net.addR("rb", 10000)

    tt = NPNTransistor(None, "", 1e-12, 25e-3, 100, 10)
    t1 = net.addComp("T1", tt)

    connect(v1.p, t1.C)
    connect(vb.p, rb.p)
    connect(rb.n, t1.B)
    connect(vb.n, net.ground)
    connect(t1.E, re.p)
    connect(re.n, v1.n)
    connect(v1.n, net.ground)
    ana = Analysis(net)
    pp.pprint(ana.analyze())

def test10():
      tt = NPNTransistor(None, "", 1e-12, 25e-3, 100, 10)
      vbe = 0.65
      vbc = -0.1
      ie = tt.IE(vbe, vbc)
      ic = tt.IC(vbe, vbc)
      ib = tt.IB(vbe, vbc)
      pp.pprint((("ie", tt.IE(vbe, vbc)),
                 ("ic", tt.IC(vbe, vbc)),
                 ("ib", tt.IB(vbe, vbc)),
                 ("d_ib_vbe", tt.d_IB_vbe(vbe)),
                 ("d_ib_vbc", tt.d_IB_vbc(vbc)),
                 ("d_ie_vbe", tt.d_IE_vbe(vbe)),
                 ("d_ie_vbc", tt.d_IE_vbc(vbc)),
                 ("d_ic_vbe", tt.d_IC_vbe(vbe)),
                 ("d_ic_vbc", tt.d_IC_vbc(vbc)),
                 ))

def test11():
    net = Network()
    cc = net.addC("cc", 0.2)
    cb = net.addC("cb", 0.02)
    re = net.addR("re", 1)
    tt = NPNTransistor(None, "", 1e-12, 25e-3, 100, 10)
    t1 = net.addComp("T1", tt)

    connect(cc.p, t1.C)
    connect(cb.p, t1.B)
    connect(cb.n, cc.n)
    connect(t1.E, re.p)
    connect(re.n, cc.n)
    connect(cc.n, net.ground)
    ana = Analysis(net)
    pp.pprint(ana.analyze())

def test12():
    net = Network()
    vc = net.addV("cc", 1.2)
    cb = net.addC("cb", 0.02)
    re = net.addR("re", 100)
    tt = NPNTransistor(None, "", 1e-16, 20e-3, 50, 0.1)
    t1 = net.addComp("T1", tt)

    connect(vc.p, t1.C)
    connect(cb.p, t1.B)
    connect(cb.n, vc.n)
    connect(t1.E, re.p)
    connect(re.n, vc.n)
    connect(vc.n, net.ground)
    ana = Analysis(net)
    pp.pprint(ana.analyze())

def test13():
    net = Network()
    vc = net.addV("vc", 5)
    vb = net.addV("vb", 2)

    rb = net.addR("rb", 1e3)
    re = net.addR("re", 100)
    tt = NPNTransistor(None, "", 1e-16, 20e-3, 50, 0.1)
    t1 = net.addComp("T1", tt)

    connect(vc.p, t1.C)
    connect(vb.p, rb.p)
    connect(rb.n,t1.B)
    connect(vb.n, vc.n)
    connect(t1.E, re.p)
    connect(re.n, vc.n)
    connect(vc.n, net.ground)
    ana = Analysis(net)
    pp.pprint(ana.analyze())


def test14():
    v = 2
    tt = NPNTransistor(None, "", 1e-12, 25e-3, 100, 10)
    nw = Network()
    net = Network()
    vc = net.addV("vc", 2)
    vb = net.addV("vb", v)
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
    res = ana.analyze(maxit=30)
    pp.pprint(res)

def test15():
    net = Network()
    vc = net.addV("vc", 2)
    r = net.addR("r", 1e2)
    c = net.addCapa("ca", 100e-6)
    connect(vc.p, c.p)
    connect(c.n, r.p)
    connect(r.n, vc.n)
    connect(vc.n, net.ground)
    ana = Analysis(net)
    res = ana.analyze(maxit=30)
    print(res)
    res = ana.analyze(maxit=30, charges={"ca": 100e-6 })
    print(res)



def main():

    test15()

main()

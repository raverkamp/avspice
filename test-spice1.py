import spice
from spice import Network, connect, compute_nodes, Analysis
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

        
        
        

def main():
    test7()




main()

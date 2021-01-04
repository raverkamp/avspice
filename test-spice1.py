import spice
from spice import Network, connect, compute_nodes, analyze
import argparse
import pprint as pp



def test1():
    net = Network()
    r1 = net.addR("r1",100)
    c1 = net.addC("c1", 1)
    connect(c1.p2, r1.p1)
    connect(r1.p2, c1.p1)
    nodes = compute_nodes(net)
    pp.pprint(nodes)

def test2():
    net = Network()
    r1 = net.addR("r1",100)
    c1 = net.addC("c1", 1)
    r2 = net.addR("r2", 200)
    connect(c1.p2, r1.p1)
    connect(r1.p2, c1.p1)
    connect(r2.p1, c1.p1)
    connect(r2.p2, c1.p2)
    nodes = compute_nodes(net)
    pp.pprint(nodes)
    n2 = analyze(net)
    pp.pprint(n2)

def test2():
    net = Network()
    r1 = net.addR("r1",100)
    c1 = net.addC("c1", 1)
    r2 = net.addR("r2", 200)
    connect(c1.p2, r1.p1)
    connect(r1.p2, c1.p1)
    connect(r2.p1, c1.p1)
    connect(r2.p2, c1.p2)
    n1 = net.addN("N1")
    connect(n1, c1.p1)
    n2 = net.addN("N2")
    connect(n2, c1.p2)
    connect(n1, net.ground)
    nodes = compute_nodes(net)
    pp.pprint(nodes)
    n2 = analyze(net)
    pp.pprint(n2)

def test3():
    net = Network()
    c1 = net.addC("c1", 1)
    connect(c1.p1, net.ground)
    r1 = net.addR("r1",10)
    r2 = net.addR("r2",20)
    r3 = net.addR("r3", 5)
    connect(c1.p2, r1.p1)
    connect(r1.p2, r2.p1)
    connect(r2.p2, r3.p1)
    connect(r3.p2, net.ground)
    n2 = analyze(net)
    pp.pprint(n2)
    xx = spice.sym_analyze(net)
    
def main():
    test3()
    
    


main()

import unittest

from spice import Network, connect, compute_nodes, Analysis, Diode, NPNTransistor, explin, dexplin

import pprint as pp
from math import exp

DIODE = Diode(None, "", 1e-8, 25e-3, 10)

class TestMath(unittest.TestCase):
    
    def test_explin(self):
        x = explin(2,3)
        self.assertAlmostEqual(x, exp(2))
        
        x = explin(3,3)
        self.assertAlmostEqual(x, exp(3))
        #
        x = explin(4,3)
        self.assertAlmostEqual(x, exp(3) + exp(3) * (4-3))

        x = explin(4,3)
        y = explin(23,3)
        self.assertAlmostEqual((y-x)/(23-4), exp(3.0))
        
        
    def test_dexplin(self):
        x = dexplin(2,3)
        self.assertAlmostEqual(x, exp(2))
        
        x = dexplin(3,3)
        self.assertAlmostEqual(x, exp(3))

        x = dexplin(4,3)
        self.assertAlmostEqual(x, exp(3))

        x = dexplin(6,3)
        self.assertAlmostEqual(x, exp(3))

        
class Test1(unittest.TestCase):

    # current and resistor
    def test_current_simple(self):
         net = Network()
         r1 = net.addR("r1", 20)
         c1 = net.addC("c1", 1)
         connect(c1.p, r1.p)
         connect(r1.n, c1.n)
         connect(r1.n, net.ground)
         analy = Analysis(net)
         (v,r,d) = analy.analyze()
         self.assertAlmostEqual(r[r1][0], 20)
         self.assertAlmostEqual(r[r1][1], 1)

    # voltage and resistor
    def test_voltage_simple(self):
         net = Network()
         r1 = net.addR("r1", 20)
         v1 = net.addV("v1", 1)
         connect(v1.p, r1.p)
         connect(r1.n, v1.n)
         connect(r1.n, net.ground)
         analy = Analysis(net)
         (v,r,d) = analy.analyze()
         self.assertAlmostEqual(r[r1][0], 1)
         self.assertAlmostEqual(r[r1][1], 1/20)

    # diode and resistor in serial
    def test_diode_simple1(self):
         net = Network()
         d1 = net.addComp("d1", DIODE)
         r1 = net.addR("r1", 500)
         v1 = net.addV("v1", 5)
         connect(v1.n, net.ground)
         connect(v1.p, d1.p)
         connect(d1.n, r1.p)
         connect(r1.n, v1.n)
         analy = Analysis(net)
         (v,r,d) = analy.analyze()
         pp.pprint((v,r,d))
         # check current is the same
         self.assertAlmostEqual(r[r1][1], d[d1][1])

   # diode - diode - resistor
    def test_diode_simple2(self):
         net = Network()
         d1 = net.addComp("d1", DIODE)
         d2 = net.addComp("d2", DIODE)
         r1 = net.addR("r1", 500)
         v1 = net.addV("v1", 5)
         connect(v1.n, net.ground)
         connect(v1.p, d1.p)
         connect(d1.n, d2.p)
         connect(d2.n, r1.p)
         connect(r1.n, v1.n)
         analy = Analysis(net)
         (v,r,d) = analy.analyze()
         pp.pprint((v,r,d))
         # check current is the same
         self.assertAlmostEqual(r[r1][1], d[d1][1])
         self.assertAlmostEqual(r[r1][1], d[d2][1])
         # check voltage diff over both diodes is equal
         self.assertAlmostEqual(d[d1][0], d[d2][0])

class TestTransistor(unittest.TestCase):

    def test_transistor_formulas(self):
        tt = NPNTransistor(None, "", 1e-12, 25e-3, 100, 10)
        ie = tt.IE(0.15, -3)
        ic = tt.IC(0.15, -3)
        ib = tt.IB(0.15, 0)
        pp.pprint((ib,ic, ie))
        self.assertAlmostEqual(ic+ib-ie,0)
        
        
        


if __name__ == '__main__':
    unittest.main()

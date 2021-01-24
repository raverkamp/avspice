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

    def test_get_object(self):
        net = Network()
        r1 = net.addR("r1", 20)
        r1_= net.get_object("r1")
        self.assertEqual(r1, r1_)
        r1_p= net.get_object("r1.p")
        self.assertEqual(r1.p, r1_p)

    # current and resistor
    def test_current_simple(self):
         net = Network()
         r1 = net.addR("r1", 20)
         c1 = net.addC("c1", 1)
         connect(c1.p, r1.p)
         connect(r1.n, c1.n)
         connect(r1.n, net.ground)
         analy = Analysis(net)
         res = analy.analyze()
         self.assertAlmostEqual(res.get_current("r1.p"), 1)
         self.assertAlmostEqual(res.get_voltage("r1.p"), 20)

    # voltage and resistor
    def test_voltage_simple(self):
         net = Network()
         r1 = net.addR("r1", 20)
         v1 = net.addV("v1", 1)
         connect(v1.p, r1.p)
         connect(r1.n, v1.n)
         connect(r1.n, net.ground)
         analy = Analysis(net)
         res = analy.analyze()
         self.assertAlmostEqual(res.get_voltage("r1.p"), 1)
         self.assertAlmostEqual(res.get_current("r1.p"), 1/20)

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
         res = analy.analyze()
         # check current is the same
         self.assertAlmostEqual(res.get_current(d1.n), - res.get_current(r1.p))

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
         res = analy.analyze()
         # check current is the same
         self.assertAlmostEqual(res.get_current(r1.p), res.get_current(d1.p))
         self.assertAlmostEqual(res.get_current(r1.p), res.get_current(d2.p))
         
         # check voltage diff over both diodes is equal
         self.assertAlmostEqual(res.get_voltage(d1.p) - res.get_voltage(d1.n),
                                res.get_voltage(d2.p) - res.get_voltage(d2.n))

class TestTransistor(unittest.TestCase):

    def test_transistor_formulas(self):
        tt = NPNTransistor(None, "", 1e-12, 25e-3, 100, 10)
        # this worked: vbe = 0.15, vbc = -3
        for vbe in [x/10 -3.0 for x in range(0,60)]:
            for vbc in [x/10 -3.0 for x in range(0,60)]:
                ie = tt.IE(vbe, vbc)
                ic = tt.IC(vbe, vbc)
                ib = tt.IB(vbe, vbc)
                self.assertAlmostEqual(ic+ib-ie,0)

    def test_transistor_formulas_2(self):
        tt = NPNTransistor(None, "", 1e-12, 25e-3, 100, 10)
        # this worked: vbe = 0.15, vbc = -3
        for v in [x/10 -3.0 for x in range(0,60)]:
            ie = tt.IE(v, -v)
            ic = tt.IC(v, -v)
            ib = tt.IB(v, -v)
            self.assertAlmostEqual(ic+ib-ie,0)
                
    def test_trans1(self):
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
        res = ana.analyze()
        self.assertAlmostEqual(res.get_current(t1.B),0.02)

if __name__ == '__main__':
    unittest.main()

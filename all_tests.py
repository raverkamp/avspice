import unittest

from spice import Network, connect, compute_nodes, Analysis, Diode, NPNTransistor

import pprint as pp

DIODE = Diode(None, "", 1e-8, 25e-3, 10)

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

    def test_diode_simple(self):
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
         self.assertAlmostEqual(r[r1][1], d[d1][1])



if __name__ == '__main__':
    unittest.main()

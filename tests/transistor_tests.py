"""simple unit tests"""

import unittest
import math
import numpy as np
from avspice import Circuit, Analysis, Diode, NPNTransistor,\
    Variable, PNPTransistor, SubCircuit, PieceWiseLinearVoltage,NPNTransistorAsNPort

from avspice.util import  explin, dexplin, linear_interpolate, smooth_step, dsmooth_step, ndiff

from avspice import ncomponents

from avspice import solving

class TestNPortTransistor(unittest.TestCase):
    """test for transistor"""

    def test_transistor_formulas(self):
        tt = ncomponents.NNPNTransistor(1e-12, 25e-3, 100, 10, -40, 40)
        # this worked: vbe = 0.15, vbc = -3
        for vbe in [x/10 -3.0 for x in range(0,60)]:
            for vbc in [x/10 -3.0 for x in range(0,60)]:
                ie = tt.IE(vbe, vbc)
                ic = tt.IC(vbe, vbc)
                ib = tt.IB(vbe, vbc)
                self.assertAlmostEqual(ic+ib-ie,0)

    def test_transistor_formulas_2(self):
        tt = ncomponents.NNPNTransistor(1e-12, 25e-3, 100, 10, -40, 40)
        # this worked: vbe = 0.15, vbc = -3
        for v in [x/10 -3.0 for x in range(0,60)]:
            ie = tt.IE(v, -v)
            ic = tt.IC(v, -v)
            ib = tt.IB(v, -v)
            self.assertAlmostEqual(ic+ib-ie,0)

    def test_trans1(self):
        net = Circuit()
        net.addV("vc", 6, "vcc", "0")
        net.addV("vb", 1, "vb","0")
        net.addR("re", 10, "E", "0")
        net.addR("rb", 10e3, "vb", "B")
        net.addR("rc", 10, "vcc", "C")
        tt = NPNTransistorAsNPort("Model T1",1e-12, 25e-3, 100, 10)
        net.add_component("t1", tt, ("B", "C", "E"))

        ana = Analysis(net)
        res = ana.analyze(maxit=50)
        self.assertAlmostEqual(res.get_current("t1.C")/res.get_current("t1.B"),100,places=5)

        self.assertAlmostEqual(res.y_norm, 0)


    def test_trans1b(self):
        net = Circuit()
        net.addV("vc", 6, "vcc", "0")
        net.addV("vb", 1, "vb","0")
        net.addR("re", 10, "E", "0")
        net.addR("rb", 10e3, "vb", "B")
        net.addR("rc", 10, "vcc", "C")
        tt = NPNTransistor("Model T1",1e-12, 25e-3, 100, 10)
        net.add_component("t1", tt, ("B", "C", "E"))

        ana = Analysis(net)
        res = ana.analyze(maxit=50)
        self.assertAlmostEqual(res.get_current("t1.C")/res.get_current("t1.B"),100,places=5)

        self.assertAlmostEqual(res.y_norm, 0)

        
    def test_trans2(self):
        net = Circuit()
        net.addV("vc",Variable("vc"), "vcc", "0")
        tt = NPNTransistorAsNPort("", 1e-12, 25e-3, 100, 10)
        net.add_component("t1", tt, ("B", "C", "0"))
        net.addR("rc", 10, "vcc", "C")
        net.addR("rb", 10e3, "vcc", "B")
        ana = Analysis(net)
        sol = None
        for x in [0.01, 0.1, 0.2, 0.3, 0.5,1,2,3,5]:
            res = ana.analyze(start_solution_vec = sol, variables={"vc":x})
            sol = res.solution_vec
        # die Konstante habe ich mir ausgeben lassen
        self.assertAlmostEqual(res.get_current("t1.B"), 438.7e-6)
        self.assertAlmostEqual(res.get_current("t1.B"), res.get_current("t1.C")/100)
        self.assertAlmostEqual(res.y_norm, 0)
        
if __name__ == '__main__':
    unittest.main()

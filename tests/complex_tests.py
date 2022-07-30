"""complex unit tests"""
import unittest
from avspice import Circuit, Analysis, Diode, NPNTransistor,\
    Variable, PNPTransistor, SubCircuit, PieceWiseLinearVoltage



class TestZener(unittest.TestCase):
    """a test using a subcircuit to simulate a zender diode
       accuracy is not that important, just check that the subc ircuit stuff works"""

    def zener(self, name, v):
        d =  Diode("D", 1e-8, 25e-3)
        cutoff = 40
        d = Diode(name, 1e-8, 25e-3, lcut_off=-cutoff, rcut_off=cutoff)
        sc =  SubCircuit(("p", "n"))
        sc.add("Df", d, ("p", "n"))
        sc.add("Dr", d, ("m", "p"))
        sc.addV("v",v, "n", "m")
        return sc

    def test_1(self):
        net = Circuit()
        v = Variable("v")
        net.addV("V", v, "VCC", "0")
        z = self.zener("Z5", 5)
        net.add("Z", z, ("VCC", "0"))
        ana = Analysis(net)

        res = ana.analyze(variables={"v": -5.5})
        current_55 = res.get_current("V.p")
        self.assertTrue(current_55 < -4.5)

        res = ana.analyze(variables={"v": -3})
        current_3 = res.get_current("V.p")
        self.assertTrue(-1e-20 <current_3  < 0)

        res = ana.analyze(variables={"v": 0.1})
        current_01 = res.get_current("V.p")
        self.assertTrue(0 <current_01  < 0.0001)

        res = ana.analyze(variables={"v": 0.3})
        current_03 = res.get_current("V.p")
        self.assertTrue(0 <current_03  < 0.002)

        res = ana.analyze(variables={"v": 0.4})
        current_04 = res.get_current("V.p")
        self.assertTrue(0.05 <current_04  < 0.1)

        res = ana.analyze(variables={"v": 0.5})
        current_05 = res.get_current("V.p")
        self.assertTrue(4 <current_05  < 6)



if __name__ == '__main__':
    unittest.main()
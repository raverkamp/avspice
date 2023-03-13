"""complex unit tests"""
import unittest
from avspice import Circuit, Analysis, Diode, NPNTransistor,\
    Variable, PNPTransistor, SubCircuit, PieceWiseLinearVoltage, ZDiode,\
    FET



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


    def test_2(self):
        net = Circuit()
        v = Variable("v")
        net.addV("V", v, "VCC", "0")
        z = ZDiode("Z5", 5, 1e-8, 25e-3)
        net.add("Z", z, ("VCC", "0"))
        ana = Analysis(net)

        res = ana.analyze(variables={"v": -5.5})
        print(res)
        current_55 = res.get_current("V.p")
        self.assertTrue(current_55 < -4.5)

        res = ana.analyze(variables={"v": -3})
        current_3 = res.get_current("V.p")
        print(current_3)
        self.assertTrue(-1e-7 <current_3  < 0)

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

class TestFET(unittest.TestCase):
    "FET tests"
    def test_1(self):
        net = Circuit()
        v = 6
        net.addV("V", v, "VCC", "0")

        net.addV("VG",7, "VCG", "0")

        net.addR("R1", 100, "VCC", "D")
        net.addR("R2", 1, "S", "0")

        net.add("F", FET("F", 1), ("VCG", "D", "S"))

        ana = Analysis(net)
        res = ana.analyze()
        current = res.get_current("F.D")
        self.assertTrue(0.058 < current < 0.06)

if __name__ == '__main__':
    unittest.main()

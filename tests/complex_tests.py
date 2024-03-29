"""complex unit tests"""
import unittest
from avspice import (
    Circuit,
    Analysis,
    Diode,
    NPNTransistor,
    Variable,
    PNPTransistor,
    SubCircuit,
    PieceWiseLinearVoltage,
    ZDiode,
    FET,
    VoltageControlledVoltageSource,
    ncomponents,
    Result,
    LinearVoltageControlledVoltageSource,
    VoltageControlledCurrentSource,
)


class TestZener(unittest.TestCase):
    """a test using a subcircuit to simulate a zender diode
    accuracy is not that important, just check that the subc ircuit stuff works"""

    def zener(self, name, v):
        d = Diode("D", 1e-8, 25e-3)
        cutoff = 40
        d = Diode(name, 1e-8, 25e-3, lcut_off=-cutoff, rcut_off=cutoff)
        sc = SubCircuit(("p", "n"))
        sc.add("Df", d, ("p", "n"))
        sc.add("Dr", d, ("m", "p"))
        sc.addV("v", v, "n", "m")
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
        self.assertTrue(-1e-20 < current_3 < 0)

        res = ana.analyze(variables={"v": 0.1})
        current_01 = res.get_current("V.p")
        self.assertTrue(0 < current_01 < 0.0001)

        res = ana.analyze(variables={"v": 0.3})
        current_03 = res.get_current("V.p")
        self.assertTrue(0 < current_03 < 0.002)

        res = ana.analyze(variables={"v": 0.4})
        current_04 = res.get_current("V.p")
        self.assertTrue(0.05 < current_04 < 0.1)

        res = ana.analyze(variables={"v": 0.5})
        current_05 = res.get_current("V.p")
        self.assertTrue(4 < current_05 < 6)

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
        self.assertTrue(-1e-7 < current_3 < 0)

        res = ana.analyze(variables={"v": 0.1})
        current_01 = res.get_current("V.p")
        self.assertTrue(0 < current_01 < 0.0001)

        res = ana.analyze(variables={"v": 0.3})
        current_03 = res.get_current("V.p")
        self.assertTrue(0 < current_03 < 0.002)

        res = ana.analyze(variables={"v": 0.4})
        current_04 = res.get_current("V.p")
        self.assertTrue(0.05 < current_04 < 0.1)

        res = ana.analyze(variables={"v": 0.5})
        current_05 = res.get_current("V.p")
        self.assertTrue(4 < current_05 < 6)


class TestFET(unittest.TestCase):
    "FET tests"

    def test_1(self):
        net = Circuit()
        v = 6
        net.addV("V", v, "VCC", "0")

        net.addV("VG", 7, "VCG", "0")

        net.addR("R1", 100, "VCC", "D")
        net.addR("R2", 1, "S", "0")

        net.add("F", FET("F", 1), ("VCG", "D", "S"))

        ana = Analysis(net)
        res = ana.analyze()
        current = res.get_current("F.D")
        self.assertTrue(0.058 < current < 0.06)


class TestCurrentSource(unittest.TestCase):
    "sweep for current source"

    def test1(self):
        net = Circuit()
        v = 10
        re = 10e3
        r1 = 1e3
        r2 = 10e3
        net.addV("V", v, "V", "0")
        rl = Variable("RL")
        net.addR("RL", rl, "V", "C")
        net.addR("R1", r1, "V", "B")
        net.addR("R2", r2, "B", "0")
        net.addR("RE", re, "E", "0")
        t1 = NPNTransistor("Model T1", 1e-12, 25e-3, 100, 10)
        net.add_component("t1", t1, ("B", "C", "E"))

        # a simple current source, the voltage drop across the transsistor is 0.5 V
        # so the voltage at E should be (10 - 0.5) * re/(re+r1)
        # and current is (10 -0.5) /(re +r1)
        cu_approx = (v - 0.5) / (re + r1)

        ana = Analysis(net)
        x = 1
        while x < re * (v / (v - 0.5) - 1):
            res = ana.analyze(variables={"RL": x})
            cu = res.get_current("RL.p")
            self.assertTrue(abs(1 - cu / cu_approx) < 0.05)
            x = x * 2


class TestVoltagecontrolledVoltageSource(unittest.TestCase):
    def test_ncompo(self):
        fac = 7
        a = ncomponents.NLinearVoltageControlledVoltageSource(fac)
        self.assertAlmostEqual(a.voltage(2), 2 * fac)
        self.assertAlmostEqual(a.dvoltage(1), fac)

    def test_compo(self):
        net = Circuit()
        vcv = VoltageControlledVoltageSource("VCF")
        net.addV("V", 5, "VINp", "0")
        net.add_component("vcv", vcv, ("VINp", "0", "VOUTp", "0"))
        net.addR("r1", 7, "VOUTp", "0")

        ana = Analysis(net)
        res = ana.analyze()
        self.assertIsInstance(res, Result)
        self.assertAlmostEqual(res.get_voltage("VINp"), 5)
        self.assertAlmostEqual(res.get_voltage("VOUTp") - res.get_voltage("0"), 5)
        self.assertAlmostEqual(res.get_current("r1.p"), 5 / 7)

    def test_compo_linear(self):
        gain = 0.3
        net = Circuit()
        vcv = LinearVoltageControlledVoltageSource("VCF", gain)
        net.addV("V", 5, "VINp", "0")
        net.add_component("vcv", vcv, ("VINp", "0", "VOUTp", "0"))
        net.addR("r1", 7, "VOUTp", "0")

        ana = Analysis(net)
        res = ana.analyze()
        self.assertIsInstance(res, Result)
        self.assertAlmostEqual(res.get_voltage("VINp"), 5)
        self.assertAlmostEqual(
            res.get_voltage("VOUTp") - res.get_voltage("0"), 5 * gain
        )
        self.assertAlmostEqual(res.get_current("r1.p"), 5 * gain / 7)


class TestVoltagecontrolledCurrentSource(unittest.TestCase):
    def test_ncompo(self):
        fac = 7
        a = ncomponents.NLinearVoltageControlledCurrentSource(fac)
        self.assertAlmostEqual(a.current(2), 2 * fac)
        self.assertAlmostEqual(a.dcurrent(17), fac)

    def test_compo(self):
        net = Circuit()
        vcc = VoltageControlledCurrentSource("VCC")
        net.addV("V", 5, "VINp", "0")
        net.add_component("vcc", vcc, ("VINp", "0", "IOUTp", "0"))
        net.addR("r1", 7, "IOUTp", "0")

        ana = Analysis(net)
        res = ana.analyze()
        self.assertIsInstance(res, Result)
        self.assertAlmostEqual(res.get_voltage("VINp"), 5)
        self.assertAlmostEqual(res.get_voltage("IOUTp") - res.get_voltage("0"), 5 * 7)
        self.assertAlmostEqual(res.get_current("r1.p"), 5)


if __name__ == "__main__":
    unittest.main()

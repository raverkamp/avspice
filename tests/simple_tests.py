"""simple unit tests"""

import unittest
import math
import numpy as np
from avspice import Circuit, Analysis, Diode, NPNTransistor,\
    Variable, PNPTransistor, SubCircuit, PieceWiseLinearVoltage
from avspice.util import  explin, dexplin, linear_interpolate, smooth_step, dsmooth_step, ndiff

from avspice import ncomponents

from avspice import solving

DIODE = Diode("D", 1e-8, 25e-3, 10)

class TestMath(unittest.TestCase):
    """test of mathematics functions"""

    def test_explin(self):
        x = explin(2,-2,3)
        self.assertAlmostEqual(x, math.exp(2))

        x = explin(3,-2,3)
        self.assertAlmostEqual(x, math.exp(3))
        #
        x = explin(4,-2,3)
        self.assertAlmostEqual(x, math.exp(3) + math.exp(3) * (4-3))


        x = explin(-4,-2,3)
        self.assertAlmostEqual(x, math.exp(-2) + math.exp(-2) * (-4-(-2)))


        x = explin(4,-2,3)
        y = explin(23,-2,3)
        self.assertAlmostEqual((y-x)/(23-4), math.exp(3.0))


    def test_dexplin(self):
        x = dexplin(2,-2,3)
        self.assertAlmostEqual(x, math.exp(2))

        x = dexplin(3,-2,3)
        self.assertAlmostEqual(x, math.exp(3))

        x = dexplin(4,-2,3)
        self.assertAlmostEqual(x, math.exp(3))

        x = dexplin(6,-2,3)
        self.assertAlmostEqual(x, math.exp(3))

        x = dexplin(-6,-2,3)
        self.assertAlmostEqual(x, math.exp(-2))

        x = dexplin(-2,-2,3)
        self.assertAlmostEqual(x, math.exp(-2))

        x = dexplin(-1.5,-2,3)
        self.assertAlmostEqual(x, math.exp(-1.5))

    def test_interpolate(self):
        x = [1, 4, 5, 10]
        y = [7, 8, 20, 1]

        for (vx, vy) in zip(x,y):
            a = linear_interpolate(x, y, vx)
            self.assertAlmostEqual(a, vy)
        self.assertEqual(linear_interpolate(x, y, 0), 7)
        self.assertEqual(linear_interpolate(x, y, 11), 1)

        self.assertAlmostEqual(linear_interpolate(x, y, 2), 7 + 1/3)
        self.assertAlmostEqual(linear_interpolate(x, y, 9), 1 + 19/5)

        self.assertEqual(linear_interpolate([1],[5],0),5)
        self.assertEqual(linear_interpolate([1],[5],1),5)
        self.assertEqual(linear_interpolate([1],[5],2),5)

    def test_smoothstep(self):
        l = -3
        r = 5
        f = 1/(r - l)
        d =  smooth_step(l, r, l - 0.01)
        self.assertEqual(d,0.0)
        d =  smooth_step(l, r, r + 0.01)
        self.assertEqual(d,1.0)

        d =  smooth_step(l, r, (r+l)/2)
        self.assertEqual(d,0.5)

        d =  smooth_step(l, r, l + 1e-2)
        self.assertTrue( 3*  pow(0.9e-2*f,2)< d < 3*  pow(1.1e-2*f,2))

        d =  smooth_step(l, r, r - 1e-2)
        self.assertTrue( 3*  pow(0.9e-2*f,2)<  1-d < 3*  pow(1.1e-2*f,2))

    def test_dsmoothstep(self):
        l = -3
        r = 5
        f = 1/(r - l)

        d =  dsmooth_step(l, r, l - 0.01)
        self.assertEqual(d,0.0)
        d =  dsmooth_step(l, r, r + 0.01)
        self.assertEqual(d,0.0)

        d =  dsmooth_step(l, r, (r+l)/2)
        self.assertEqual(d,1.5*f)

        d =  dsmooth_step(l, r, l + 1e-2)
        self.assertTrue( 6*  0.9e-2 *f *f< d < 6* 1.1e-2*f*f)

        d =  dsmooth_step(l, r, r - 1e-2)
        self.assertTrue( 6 *  0.9e-2*f * f <  d < 6*  1.1e-2*f*f)

    def test_ndiff(self):
        def f(x):
            return x**3 + 5*x

        def df(x):
            return 3 * x**2 + 5

        def ndf(x):
            return ndiff(f,x)

        self.assertAlmostEqual(df(1)/ndf(1.0),1)
        self.assertAlmostEqual(df(1e10)/ndf(1e10), 1)
        self.assertAlmostEqual(df(2e-5)/ndf(2e-5), 1)


class TestSolve(unittest.TestCase):

    """test the solving code"""

    def test_simple_solve1(self):

        def f(x):
            y = np.array([x[0], x[1]])
            return y

        def Df(x):
            df = np.zeros((2,2))
            df[0][0] = 1
            df[1][1] = 1
            return df

        (sol, y, dfx, iterations, norm_y) = solving.solve(np.array([1,2]), f, Df, 1e-8, 1e-8)
        self.assertAlmostEqual(sol[0], 0)
        self.assertAlmostEqual(sol[1], 0)

    def test_simple_solve2(self):

        def f(x):
            y = np.array([x[0]*x[0] - 4])
            return y
        def Df(x):
            df = np.zeros((1,1))
            df[0][0] = 2 * x[0]
            return df

        (sol, y, dfx, iterations, norm_y) = solving.solve(np.array([1]), f, Df, 1e-8, 1e-8)
        self.assertAlmostEqual(sol[0], 2)

class NetworkContstruct(unittest.TestCase):

    def test_init(self):
        n = Circuit()
        n.addR("r",20, "0", "1")
        n.addC("c0",40, "1", "2")
        n.addV("c1",40, "2", "3")
        n.addD("c2",20,23, "4", "5")
        n.addCapa("c3",40, "2", "3")
        n.addInduc("c4",40, "2", "3")

        t1 = PNPTransistor("t1", 1,2,3,4)
        n.add_component("t1", t1, ("0", "1", "2"))

        t2 = PNPTransistor("t1", 1,2,3,4)
        n.add_component("t2", t2, ("0", "1", "2"))

class Test1(unittest.TestCase):

    # current and resistor
    def test_current_simple(self):
        net = Circuit()
        net.addR("r1", 20, "1", "0")
        net.addC("c1", 1, "1", "0")

        analy = Analysis(net)
        res = analy.analyze()

        res.display()

        self.assertAlmostEqual(res.get_current("r1.p"), 1)
        self.assertAlmostEqual(res.get_voltage("r1.p"), 20)
        self.assertAlmostEqual(res.y_norm, 0)

    def test_current_simple1b(self):
        net = Circuit()
        net.addR("r1", 20, "1", "0")
        var = Variable("v1")
        net.addC("c1", var, "1", "0")

        analy = Analysis(net)
        res = analy.analyze(variables={"v1":1})

        res.display()

        self.assertAlmostEqual(res.get_current("r1.p"), 1)
        self.assertAlmostEqual(res.get_voltage("r1.p"), 20)
        self.assertAlmostEqual(res.y_norm, 0)


    def test_current_simple2(self):
        #parallel current source
        net = Circuit()
        net.addR("r1", 20,  "1", "0")
        net.addC("c1", 1, "1", "0")
        net.addC("c2", 2, "1", "0")
        analy = Analysis(net)
        res = analy.analyze()
        self.assertAlmostEqual(res.get_current("r1.p"), 3)
        self.assertAlmostEqual(res.get_voltage("r1.p"), 60)
        self.assertAlmostEqual(res.y_norm, 0)

    # voltage and resistor
    def test_voltage_simple(self):
        net = Circuit()
        net.addR("r1", 20, "1", "0")
        net.addV("v1", 1, "1", "0")
        analy = Analysis(net)
        res = analy.analyze()
        self.assertAlmostEqual(res.get_voltage("r1.p"), 1)
        self.assertAlmostEqual(res.get_current("r1.p"), 1/20)
        self.assertAlmostEqual(res.y_norm, 0)

    def test_voltage_simpleb(self):
        net = Circuit()
        var = Variable("v1")
        net.addR("r1", 20, "1", "0")
        net.addV("v1", var, "1", "0")
        analy = Analysis(net)
        x = 5
        res = analy.analyze(variables={"v1": x})
        self.assertAlmostEqual(res.get_voltage("r1.p"), x)
        self.assertAlmostEqual(res.get_current("r1.p"), x/20)
        self.assertAlmostEqual(res.y_norm, 0)


    # diode and resistor in serial
    def test_diode_simple1(self):
        net = Circuit()
        net.add_component("d1", DIODE, ("vcc", "r"))
        net.addR("r1", 500, "r", "0")
        net.addV("v1", 5, "vcc", "0")
        analy = Analysis(net)
        res = analy.analyze(abstol=1e-12, reltol= 1e-12, maxit=20)
        # check current is the same
        self.assertAlmostEqual(res.get_current("d1.n"), -res.get_current("r1.p"))
        self.assertAlmostEqual(res.y_norm, 0)

   # diode - diode - resistor
    def test_diode_simple2(self):
        net = Circuit()
        net.add_component("d1", DIODE, ("vcc", "d1"))
        net.add_component("d2", DIODE, ("d1", "d2"))
        net.addR("r1", 500, "d2","0")
        net.addV("v1", 5, "vcc", "0")
        analy = Analysis(net)
        res = analy.analyze(abstol=1e-10,reltol=1e-8)
        # check current is the same
        self.assertAlmostEqual(res.get_current("r1.p"), res.get_current("d1.p"))
        self.assertAlmostEqual(res.get_current("r1.p"), res.get_current("d2.p"))

        # check voltage diff over both diodes is equal
        self.assertAlmostEqual(res.get_voltage("d1.p") - res.get_voltage("d1.n"),
                                res.get_voltage("d2.p") - res.get_voltage("d2.n"))
        self.assertAlmostEqual(res.y_norm, 0)

    # diode|diode -> resistor
    def test_diode_simple3(self):
        net = Circuit()
        net.add_component("d1", DIODE, ("vcc","d"))
        net.add_component("d2", DIODE, ("vcc", "d"))
        net.addR("r1", 500, "d", "0")
        net.addV("v1", 5, "vcc", "0")
        analy = Analysis(net)
        res = analy.analyze()
        # check current is the same over both diodes
        self.assertAlmostEqual(res.get_current("r1.p")/2, res.get_current("d1.p"))
        self.assertAlmostEqual(res.get_current("r1.p")/2, res.get_current("d2.p"))
        self.assertAlmostEqual(res.y_norm, 0)

    def test_voltage(self):
        net = Circuit()
        net.addV("v1", 3, "v1p", "v2p")
        net.addV("v2", 5, "v2p", "v3p")
        net.addV("v3", 7, "v3p", "0")
        net.addR("r1", 1000, "v1p", "0")
        net.addR("r2", 100, "v1p", "v3p")

        analy = Analysis(net)
        res = analy.analyze()
        self.assertAlmostEqual(res.get_current("r1.p"),(3+5+7)/1000)
        self.assertAlmostEqual(res.get_current("r2.p"),(3+5)/100)
        self.assertAlmostEqual(res.y_norm, 0)




    def test_var(self):
        net = Circuit()
        volt_var = Variable("v")
        net.addV("v1", volt_var, "vcc","0")
        net.addR("r2", 100, "vcc", "0")


        analy = Analysis(net)
        res = analy.analyze(variables={"v":5})
        self.assertAlmostEqual(res.get_current("r2.p"),5/100)
        res = analy.analyze(variables={"v":6})
        self.assertAlmostEqual(res.get_current("r2.p"), 6/100)
        self.assertAlmostEqual(res.y_norm, 0)

    def test_var2(self):
        """test default for variable"""
        net = Circuit()
        volt_var = Variable("v", 5)
        net.addV("v1", volt_var, "vcc", "0")
        net.addR("r2", 100, "vcc", "0")

        analy = Analysis(net)
        res = analy.analyze()
        self.assertAlmostEqual(res.get_current("r2.p"),5/100)
        res = analy.analyze(variables={"v":6})
        self.assertAlmostEqual(res.get_current("r2.p"), 6/100)
        self.assertAlmostEqual(res.y_norm, 0)



class TestTransistor(unittest.TestCase):
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
        tt = NPNTransistor("Model T1",1e-12, 25e-3, 100, 10)
        net.add_component("t1", tt, ("B", "C", "E"))

        ana = Analysis(net)
        res = ana.analyze(maxit=50)
        self.assertAlmostEqual(res.get_current("t1.C")/res.get_current("t1.B"),100,places=5)

        self.assertAlmostEqual(res.y_norm, 0)

    def test_trans2(self):
        net = Circuit()
        net.addV("vc",Variable("vc"), "vcc", "0")
        tt = NPNTransistor("", 1e-12, 25e-3, 100, 10)
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

    def test_trans_fw(self):

        beta_f = 100
        beta_r = 20
        tt = NPNTransistor("T", 1e-12, 25e-3, beta_f, beta_r)

        net = Circuit()
        net.addV("v", 6, "vcc", "0")
        net.addR("re", 10, "E", "0")
        net.addR("rb", 20e3, "vcc", "B")
        net.addR("rc", 10, "vcc", "C")
        net.add_component("t1", tt, ("B", "C", "E"))

        ana = Analysis(net)
        res = ana.analyze(maxit=50)
        self.assertTrue(not isinstance(res, str), "could not find solution: {res}")
        self.assertTrue(0.8 <= res.get_current("t1.B")/(6/20e3) <= 1.2,
                        f"wrong base current: {res.get_current('t1.B')}")
        self.assertAlmostEqual(-res.get_current("t1.E")/beta_f,res.get_current("t1.B"),places=5)
        self.assertAlmostEqual(res.y_norm, 0)

    def test_trans_bw(self):

        beta_f = 100
        beta_r = 20
        tt = NPNTransistor("", 1e-12, 25e-3, beta_f, beta_r)

        net = Circuit()
        net.addV("v", 6, "vcc", "0")
        net.addR("re", 10, "C", "0")
        net.addR("rb", 20e3, "vcc", "B")
        net.addR("rc", 10, "vcc", "E")
        net.add_component("t1", tt, ("B", "C", "E"))

        ana = Analysis(net)
        res = ana.analyze(maxit=50)
        self.assertTrue(not isinstance(res, str), "could not find solution: {res}")
        self.assertTrue(0.8 <= res.get_current("t1.B")/(6/20e3) <= 1.2, \
                        f"wrong base current: {res.get_current('t1.B')}")
        self.assertAlmostEqual(res.get_current("t1.E")/beta_r,res.get_current("t1.B"),places=5)
        self.assertAlmostEqual(res.y_norm, 0)


class PNPTransistorTests(unittest.TestCase):
    """test for PNPtransistor"""

    def test_formulas(self):
        tt = ncomponents.NPNPTransistor(1e-12, 25e-3, 100, 10, -40, 40)


        for vbc in [-0.3, -0.1, 0, 0.1, 0.3]:
            for vbe in [-0.3, -0.1, 0, 0.1, 0.3]:

                ie = tt.IE(vbe, vbc)
                ic = tt.IC(vbe, vbc)
                ib = tt.IB(vbe, vbc)

                #print(ie, ic, ib)

                self.assertAlmostEqual(ie+ ic + ib,0)

                d_vbe = tt.d_IB_vbe(vbe) + tt.d_IC_vbe(vbe) + tt.d_IE_vbe(vbe)

                self.assertAlmostEqual(d_vbe,0)

                d_vbc = tt.d_IB_vbc(vbc) + tt.d_IC_vbc(vbc) + tt.d_IE_vbc(vbc)

                self.assertAlmostEqual(d_vbc,0)

        ie = tt.IE(-0.3,0.2)
        self.assertGreater(ie,0)

        ic = tt.IC(0.2,-0.3)
        self.assertGreater(ic,0)

    def test_nformulas(self):
        tt = ncomponents.NPNPTransistor(1e-12, 25e-3, 100, 10)


        for vbc in [-0.3, -0.1, 0, 0.1,0.3]:
            for vbe in [-0.3, -0.1, 0, 0.1,0.3]:

                ie = tt.IE(vbe, vbc)
                ic = tt.IC(vbe, vbc)
                ib = tt.IB(vbe, vbc)

                #print(ie, ic, ib)

                self.assertAlmostEqual(ie+ ic + ib,0)

                d_vbe = tt.d_IB_vbe(vbe) + tt.d_IC_vbe(vbe) + tt.d_IE_vbe(vbe)

                self.assertAlmostEqual(d_vbe,0)

                d_vbc = tt.d_IB_vbc(vbc) + tt.d_IC_vbc(vbc) + tt.d_IE_vbc(vbc)

                self.assertAlmostEqual(d_vbc,0)

        ie = tt.IE(-0.3,0.2)
        self.assertGreater(ie,0)

        ic = tt.IC(0.2,-0.3)
        self.assertGreater(ic,0)


    def test_trans_diode(self):
        beta_f = 100
        beta_r = 20
        tt = PNPTransistor("T",1e-12, 25e-3, beta_f, beta_r)
        ntt = ncomponents.NPNPTransistor(1e-12, 25e-3, beta_f, beta_r, -40, 40)
        v0 = 6
        r0 = 20e3

        def f(vm):
            ib = -ntt.IB(-vm, -vm)
            ir = (v0 -vm) / r0
            return ib -ir

        vv = solving.bisect(f, 0.001, v0)

        net = Circuit()
        net.addV("v", v0, "vcc", "0")
        net.addR("r", r0, "vcc", "X")

        net.add_component("t1", tt, ("0", "X", "X"))
        ana = Analysis(net)
        res = ana.analyze(maxit=50)
        self.assertTrue(not isinstance(res,str), f"no solution: {res}")
        self.assertAlmostEqual(res.get_current("r.p"), (v0-vv)/r0)

    def test_trans_fw(self):

        beta_f = 100
        beta_r = 20
        tt = PNPTransistor("", 1e-12, 25e-3, beta_f, beta_r)

        net = Circuit()
        net.addV("v", 6, "vcc", "re")
        net.addR("re", 10, "re", "E")
        net.addR("rb", 20e3, "B", "0")
        net.addR("rb2", 20e3, "vcc", "B")
        net.addR("rc", 10, "C", "0")
        net.add_component("t1", tt, ("B", "C", "E"))

        ana = Analysis(net)
        res = ana.analyze(maxit=50)
        self.assertTrue(not isinstance(res,str), f"no solution {res}")
        self.assertAlmostEqual(-res.get_current("t1.E")/beta_f,res.get_current("t1.B"),places=5)
        self.assertAlmostEqual(res.y_norm, 0)


    def test_trans_bw(self):

        beta_f = 100
        beta_r = 20
        tt = PNPTransistor("", 1e-12, 25e-3, beta_f, beta_r)

        net = Circuit()
        net.addV("v", 6, "vcc", "0")
        net.addR("re", 10,"vcc", "C")
        net.addR("rb", 20e3, "B", "0")
        net.addR("rb2", 20e3, "vcc", "B")
        net.addR("rc", 10, "E", "0")
        net.add_component("t1", tt, ("B", "C", "E"))

        ana = Analysis(net)
        res = ana.analyze(maxit=50)

        self.assertTrue(not isinstance(res,str), f"no solution {res}")

        self.assertAlmostEqual(res.get_current("t1.E")/beta_r,res.get_current("t1.B"),places=5)
        self.assertAlmostEqual(res.y_norm, 0)

class ResultDisplayTest(unittest.TestCase):

    def test_result_display(self):
        npntransistor = NPNTransistor("T", 1e-12, 25e-3, 100, 10)
        net = Circuit()
        net.addV("vc", 5, "v", "0")
        net.addV("vb", Variable("vb", 2), "vb", "0")
        net.addR("rc", 100, "v", "C")
        net.addR("rb", 1e3, "vb", "B")
        net.add_component("t1", npntransistor, ("B", "C", "0"))
        ana = Analysis(net)
        res = ana.analyze(maxit=50)
        self.assertTrue(not isinstance(res,str), f"no solution {res}")

        res.display()

class TransientTest(unittest.TestCase):

    def test1(self):

        v0 = 5.6
        r = 17e3
        capa = 312e-6
        timespan = 1
        net = Circuit()
        net.addV("vc", v0, "v", "0")
        net.addR("rc", r, "v", "c")
        net.addCapa("ca", capa, "c", "0")


        ana = Analysis(net)
        # assumption capacitor is empty and will be loaded
        res = ana.transient(timespan,0.01, capa_voltages={"ca" : 0})

        ve = res.get_voltage("ca.p")[-1]
        ve_expected = v0 *(1-math.exp(-timespan/(r*capa)))
        print(ve, ve_expected)
        self.assertTrue(0.98 < ve/ve_expected <1.02)

    def test1b(self):
        r = 1000000
        capa = 1e-6
        timespan = 1
        net = Circuit()
        net.addR("rc", r, "v", "0")
        net.addCapa("ca", capa, "v", "0")

        ana = Analysis(net)
        # assumption capacitor has voltage 1
        res = ana.transient(timespan,0.01, capa_voltages={"ca" : 1})

        ve = res.get_voltage("ca.p")[-1]
        ve_expected =  math.exp(-timespan/(r*capa))
        print(ve, ve_expected)
        self.assertTrue(0.98 < ve/ve_expected <1.02)

    def test2(self):
        net = Circuit()
        v0 = 10
        net.addSineV("vc", v0, 1, "v", "0")
        net.addR("rc", 100, "v", "0")
        ana = Analysis(net)
        res = ana.transient(1,0.005)

        for (t, v) in zip(res.get_time(), res.get_voltage("rc.p")):
            self.assertAlmostEqual(v, v0 * math.sin(2* math.pi * t))

    def test3(self):
        net = Circuit()
        ro = 1e1
        indo =  10
        curro = 1
        net.addR("rc", ro, "1", "0")
        net.addInduc("ind", indo, "1", "0")
        ana = Analysis(net)
        res = ana.transient(1,0.01, induc_currents={"ind": curro})


        for (t, curr) in zip(res.get_time(), res.get_current("ind.p")):
            curr_expected = math.exp(-t*ro/indo)*curro
            self.assertTrue(0.98 < curr/curr_expected <1.02)


class TestInductor(unittest.TestCase):

    def test1(self):
        net = Circuit()
        v0 = 10
        net.addV("vc", v0, "v", "0")
        net.addR("r1", 75, "v", "in")
        net.addR("r2", 25, "r2", "0")
        net.addInduc("ind", 100, "in", "r2")
        ana = Analysis(net)
        res = ana.analyze()
        self.assertAlmostEqual(res.get_voltage("ind.p"), res.get_voltage("ind.n"))
        self.assertAlmostEqual(res.get_current("ind.p"), res.get_current("r1.p"))
        self.assertAlmostEqual(res.get_current("ind.p"), res.get_current("r2.p"))
        self.assertAlmostEqual(res.get_voltage("ind.p"), 2.5)

    def test2(self):
        net = Circuit()
        v0 = 10
        net.addV("vc", v0, "v", "0")
        net.addR("r1", 75, "v", "in")
        net.addR("r2", 25, "r2", "0")
        net.addInduc("ind", 100, "in", "r2")
        ana = Analysis(net)
        res = ana.analyze(induc_currents={"ind": 0.09},transient=True)
        #self.assertAlmostEqual(res.get_voltage(ind.p), res.get_voltage(ind.n))
        self.assertAlmostEqual(res.get_current("ind.p"), res.get_current("r1.p"))
        self.assertAlmostEqual(res.get_current("ind.p"), res.get_current("r2.p"))
        self.assertAlmostEqual(res.get_current("ind.p"), 0.09)

class TestSubCircuit(unittest.TestCase):

    def test_construct(self):
        s = SubCircuit(("A", "B"))
        s.addR("r1",10, "A","c")
        s.addR("r2",90, "c","B")

        c = Circuit()
        c.add_subcircuit("sub", s, ("VCC", "0"))
        c.addV("V", 10, "VCC", "0")

        ana = Analysis(c)
        res = ana.analyze()
        res.display()
        self.assertAlmostEqual(res.get_current("sub/r1.p"), 0.1)

    def test_darlington(self):

        sc = SubCircuit(("B", "C", "E"))
        tt = NPNTransistor("", 1e-12, 25e-3, 100, 20)
        sc.add_component("t1", tt, ("B", "C", "B2"))
        sc.add_component("t2", tt, ("B2", "C", "E"))

        c = Circuit()
        c.addV("VCC", 10, "vcc", "0")
        c.addR("RB", 10e3, "vcc", "B")


        c.add_subcircuit("dar", sc, ("B", "vcc", "0"))

        ana = Analysis(c)
        res = ana.analyze()
        self.assertTrue(abs(res.get_current("dar/t2.E")/(-8.8) -1) < 0.001)

class TestVoltageTransient(unittest.TestCase):

    def test_stepwise(self):
        """test for PieceWiseLinearVoltage"""
        net = Circuit()
        v = PieceWiseLinearVoltage("name", [(0.5,0), (1,1), (4,0)])
        net.add_component("V", v, ("VCC", "0"))
        net.addR("R1",90,"VCC", "1")
        net.addR("R2",10,"1", "0")
        ana = Analysis(net)
        res = ana.transient(5,0.01)

        def vat(t):
            return res.get_voltage_at(t, "R1.n")
        def cat(t):
            return res.get_current_at(t, "R1.p")

        self.assertAlmostEqual(vat(0.1), 0)
        self.assertAlmostEqual(cat(0.1),0)


        self.assertAlmostEqual(vat(0.75), 1/2/10)
        self.assertAlmostEqual(cat(0.75),0.5/100)

        self.assertAlmostEqual(vat(3), 0.1*(1/3 *1 + 2/3*0))
        self.assertAlmostEqual(cat(3), 1/300)



if __name__ == '__main__':
    unittest.main()

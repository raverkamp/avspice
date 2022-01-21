"""simple unit tests"""
import unittest
import pprint as pp
from math import exp
import numpy as np
from spice import Network, connect, Analysis, Diode, NPNTransistor, explin, dexplin, Variable, PNPTransistor

import solving

DIODE = Diode(None, "", 1e-8, 25e-3, 10)

class TestMath(unittest.TestCase):
    """test of mathematics functions"""

    def test_explin(self):
        x = explin(2,-2,3)
        self.assertAlmostEqual(x, exp(2))

        x = explin(3,-2,3)
        self.assertAlmostEqual(x, exp(3))
        #
        x = explin(4,-2,3)
        self.assertAlmostEqual(x, exp(3) + exp(3) * (4-3))


        x = explin(-4,-2,3)
        self.assertAlmostEqual(x, exp(-2) + exp(-2) * (-4-(-2)))


        x = explin(4,-2,3)
        y = explin(23,-2,3)
        self.assertAlmostEqual((y-x)/(23-4), exp(3.0))


    def test_dexplin(self):
        x = dexplin(2,-2,3)
        self.assertAlmostEqual(x, exp(2))

        x = dexplin(3,-2,3)
        self.assertAlmostEqual(x, exp(3))

        x = dexplin(4,-2,3)
        self.assertAlmostEqual(x, exp(3))

        x = dexplin(6,-2,3)
        self.assertAlmostEqual(x, exp(3))

        x = dexplin(-6,-2,3)
        self.assertAlmostEqual(x, exp(-2))

        x = dexplin(-2,-2,3)
        self.assertAlmostEqual(x, exp(-2))

        x = dexplin(-1.5,-2,3)
        self.assertAlmostEqual(x, exp(-1.5))


class TestSolve(unittest.TestCase):

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


class Test1(unittest.TestCase):
    """more tests"""

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
        self.assertAlmostEqual(res.y_norm, 0)

    def test_current_simple2(self):
        #parallel current source
        net = Network()
        r1 = net.addR("r1", 20)
        c1 = net.addC("c1", 1)
        c2 = net.addC("c2", 2)
        connect(c1.p, c2.p)
        connect(c1.n, c2.n)
        connect(c1.p, r1.p)
        connect(r1.n, c1.n)
        connect(r1.n, net.ground)
        analy = Analysis(net)
        res = analy.analyze()
        self.assertAlmostEqual(res.get_current("r1.p"), 3)
        self.assertAlmostEqual(res.get_voltage("r1.p"), 60)
        self.assertAlmostEqual(res.y_norm, 0)

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
        self.assertAlmostEqual(res.y_norm, 0)

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
        res = analy.analyze(abstol=1e-12, reltol= 1e-12, maxit=20)
        # check current is the same
        self.assertAlmostEqual(res.get_current(d1.n), -res.get_current(r1.p))
        self.assertAlmostEqual(res.y_norm, 0)

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
        res = analy.analyze(abstol=1e-10,reltol=1e-8)
        # check current is the same
        self.assertAlmostEqual(res.get_current(r1.p), res.get_current(d1.p))
        self.assertAlmostEqual(res.get_current(r1.p), res.get_current(d2.p))

        # check voltage diff over both diodes is equal
        self.assertAlmostEqual(res.get_voltage(d1.p) - res.get_voltage(d1.n),
                                res.get_voltage(d2.p) - res.get_voltage(d2.n))
        self.assertAlmostEqual(res.y_norm, 0)

    # diode|diode -> resistor
    def test_diode_simple3(self):
        net = Network()
        d1 = net.addComp("d1", DIODE)
        d2 = net.addComp("d2", DIODE)
        r1 = net.addR("r1", 500)
        v1 = net.addV("v1", 5)
        connect(v1.n, net.ground)
        connect(v1.p, d1.p)
        connect(v1.p, d2.p)
        connect(d1.n, r1.p)
        connect(d2.n, r1.p)
        connect(r1.n, v1.n)
        analy = Analysis(net)
        res = analy.analyze()
        # check current is the same over both diodes
        self.assertAlmostEqual(res.get_current(r1.p)/2, res.get_current(d1.p))
        self.assertAlmostEqual(res.get_current(r1.p)/2, res.get_current(d2.p))
        self.assertAlmostEqual(res.y_norm, 0)

    def test_voltage(self):
        net = Network()
        v1 = net.addV("v1", 3)
        v2 = net.addV("v2", 5)
        v3 = net.addV("v3", 7)
        r1 = net.addR("r1", 1000)
        r2 = net.addR("r2", 100)

        connect(r2.p, v1.p)
        connect(r2.n, v2.n)

        connect(v1.p,r1.p)
        connect(v1.n,v2.p)
        connect(v2.n,v3.p)
        connect(v3.n,r1.n)
        connect(r1.n, net.ground)
        analy = Analysis(net)
        res = analy.analyze()
        self.assertAlmostEqual(res.get_current(r1.p),(3+5+7)/1000)
        self.assertAlmostEqual(res.get_current(r2.p),(3+5)/100)
        self.assertAlmostEqual(res.y_norm, 0)




    def test_var(self):
        net = Network()
        volt_var = Variable("v")
        v = net.addV("v1", volt_var)
        r = net.addR("r2", 100)

        connect(v.p, r.p)
        connect(v.n, r.n)

        connect(r.n, net.ground)
        analy = Analysis(net)
        res = analy.analyze(variables={"v":5})
        self.assertAlmostEqual(res.get_current(r.p),5/100)
        res = analy.analyze(variables={"v":6})
        self.assertAlmostEqual(res.get_current(r.p), 6/100)
        self.assertAlmostEqual(res.y_norm, 0)

    def test_var2(self):
        """test default for variable"""
        net = Network()
        volt_var = Variable("v", 5)
        v = net.addV("v1", volt_var)
        r = net.addR("r2", 100)

        connect(v.p, r.p)
        connect(v.n, r.n)

        connect(r.n, net.ground)
        analy = Analysis(net)
        res = analy.analyze()
        self.assertAlmostEqual(res.get_current(r.p),5/100)
        res = analy.analyze(variables={"v":6})
        self.assertAlmostEqual(res.get_current(r.p), 6/100)
        self.assertAlmostEqual(res.y_norm, 0)



class TestTransistor(unittest.TestCase):
    """test for transistor"""

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
        vc = net.addV("vc", 6)
        vb = net.addV("vb", 1)
        re = net.addR("re", 10)
        rb = net.addR("rb", 10e3)
        rc = net.addR("rc", 10)
        tt = NPNTransistor(None, "", 1e-12, 25e-3, 100, 10)
        t1 = net.addComp("T1", tt)

        connect(vc.p, rc.p)
        connect(rc.n, t1.C)
        connect(vb.p, rb.p)
        connect(rb.n, t1.B)
        connect(vb.n, net.ground)
        connect(t1.E, re.p)
        connect(re.n, vc.n)
        connect(vc.n, net.ground)
        ana = Analysis(net)
        res = ana.analyze(maxit=50)
        self.assertAlmostEqual(res.get_current(t1.C)/res.get_current(t1.B),100,places=5)

        self.assertAlmostEqual(res.y_norm, 0)

    def test_trans2(self):
        net = Network()
        vc = net.addV("vc",Variable("vc"))
        tt = NPNTransistor(None, "", 1e-12, 25e-3, 100, 10)
        t1 = net.addComp("T1", tt)
        rc = net.addR("rc", 10)
        rb = net.addR("rb", 10e3)
        connect(vc.p, rc.p)
        connect(vc.p, rb.p)
        connect(vc.n, net.ground)
        connect(rc.n, t1.C)
        connect(rb.n, t1.B)
        connect(t1.E, net.ground)
        ana = Analysis(net)
        sol = None
        for x in [0.01, 0.1, 0.2, 0.3, 0.5,1,2,3,5]:
            res = ana.analyze(start_solution_vec = sol, variables={"vc":x})
            sol = res.solution_vec
        # die Konstante habe ich mir ausgeben lassen
        self.assertAlmostEqual(res.get_current(t1.B), 438.7e-6)
        self.assertAlmostEqual(res.get_current(t1.B), res.get_current(t1.C)/100)
        self.assertAlmostEqual(res.y_norm, 0)

    def test_trans3(self):
        # use neenergy_ levels to find solution
        net = Network()
        vc = net.addV("vc",Variable("vc", 5))
        tt = NPNTransistor(None, "", 1e-12, 25e-3, 100, 10)
        t1 = net.addComp("T1", tt)
        rc = net.addR("rc", 10)
        rb = net.addR("rb", 10e3)
        connect(vc.p, rc.p)
        connect(vc.p, rb.p)
        connect(vc.n, net.ground)
        connect(rc.n, t1.C)
        connect(rb.n, t1.B)
        connect(t1.E, net.ground)
        ana = Analysis(net)
        sol = None
        res = ana.analyze(start_solution_vec = None, abstol=1e-8, reltol = 1e-9)
        # die Konstante habe ich mir ausgeben lassen
        self.assertAlmostEqual(res.get_current(t1.B), 438.7e-6)
        self.assertAlmostEqual(res.get_current(t1.B), res.get_current(t1.C)/100)
        self.assertAlmostEqual(res.y_norm, 0)

    def test_trans_fw(self):

        beta_f = 100
        beta_r = 20
        tt = NPNTransistor(None, "", 1e-12, 25e-3, beta_f, beta_r)

        net = Network()
        v = net.addV("v", 6)
        re = net.addR("re", 10)
        rb = net.addR("rb", 20e3)
        rc = net.addR("rc", 10)
        t1 = net.addComp("T1", tt)

        connect(v.p, rc.p)
        connect(rc.n, t1.C)
        connect(rb.p, v.p)
        connect(rb.n, t1.B)
        connect(t1.E, re.p)
        connect(re.n, v.n)
        connect(v.n, net.ground)
        ana = Analysis(net)
        res = ana.analyze(maxit=50)
        self.assertTrue(not isinstance(res, str), "could not find solution: {0}".format(res))
        self.assertTrue(0.8 <= res.get_current(t1.B)/(6/20e3) <= 1.2, "wrong base current: {0}".format(res.get_current(t1.B)))
        self.assertAlmostEqual(-res.get_current(t1.E)/beta_f,res.get_current(t1.B),places=5)
        self.assertAlmostEqual(res.y_norm, 0)

    def test_trans_bw(self):

        beta_f = 100
        beta_r = 20
        tt = NPNTransistor(None, "", 1e-12, 25e-3, beta_f, beta_r)

        net = Network()
        v = net.addV("v", 6)
        re = net.addR("re", 10)
        rb = net.addR("rb", 20e3)
        rc = net.addR("rc", 10)
        t1 = net.addComp("T1", tt)

        connect(v.p, rc.p)
        connect(rc.n, t1.E)
        connect(rb.p, v.p)
        connect(rb.n, t1.B)
        connect(t1.C, re.p)
        connect(re.n, v.n)
        connect(v.n, net.ground)
        ana = Analysis(net)
        res = ana.analyze(maxit=50)
        self.assertTrue(not isinstance(res, str), "could not find solution: {0}".format(res))
        self.assertTrue(0.8 <= res.get_current(t1.B)/(6/20e3) <= 1.2, "wrong base current: {0}".format(res.get_current(t1.B)))
        self.assertAlmostEqual(res.get_current(t1.E)/beta_r,res.get_current(t1.B),places=5)
        self.assertAlmostEqual(res.y_norm, 0)


class PNPTransistorTests(unittest.TestCase):
    """test for PNPtransistor"""

    def test_formulas(self):
        tt = PNPTransistor(None, "", 1e-12, 25e-3, 100, 10)


        for vbc in [-0.3, -0.1, 0, 0.1,0.3]:
            for vbe in [-0.3, -0.1, 0, 0.1,0.3]:

                ie = tt.IE(vbe, vbc)
                ic = tt.IC(vbe, vbc)
                ib = tt.IB(vbe, vbc)

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
        tt = PNPTransistor(None, "", 1e-12, 25e-3, beta_f, beta_r)

        v0 = 6
        r0 = 20e3

        def f(vm):
            ib = -tt.IB(-vm, -vm)
            ir = (v0 -vm) / r0
            return ib -ir

        vv = solving.bisect(f, 0.001, v0)

        net = Network()
        v = net.addV("v", v0)
        r = net.addR("r", r0)

        t1 = net.addComp("t1", tt)

        connect(v.p, r.p)
        connect(r.n, t1.C)
        connect(r.n, t1.E)
        connect(t1.B, v.n)
        connect(net.ground, v.n)
        ana = Analysis(net)
        res = ana.analyze(maxit=50)
        if isinstance(res, str):
            raise Exception(res)
        #print("current = {0}, dvolt) = {1}".format(res.get_current("r.p"), res.get_voltage("t1.E")))
        #print(" bisect i={0}, v={1}".format((v0-vv)/r0, vv))
        self.assertAlmostEqual(res.get_current("r.p"), (v0-vv)/r0)

    def test_trans_fw(self):

        beta_f = 100
        beta_r = 20
        tt = PNPTransistor(None, "", 1e-12, 25e-3, beta_f, beta_r)

        net = Network()
        v = net.addV("v", 6)
        re = net.addR("re", 10)
        rb = net.addR("rb", 20e3)
        rb2 = net.addR("rb2", 20e3)
        rc = net.addR("rc", 10)
        t1 = net.addComp("T1", tt)

        connect(v.p, re.p)
        connect(re.n, t1.E)
        connect(rb.n, net.ground)
        connect(rb.p, t1.B)
        connect(rb2.n, t1.B)
        connect(rb2.p, v.p)

        connect(t1.C, rc.p)
        connect(rc.n, v.n)
        connect(v.n, net.ground)
        ana = Analysis(net)
        res = ana.analyze(maxit=50)
        if isinstance(res, str):
            raise Exception(res)
        self.assertAlmostEqual(-res.get_current(t1.E)/beta_f,res.get_current(t1.B),places=5)
        self.assertAlmostEqual(res.y_norm, 0)


    def test_trans_bw(self):

        beta_f = 100
        beta_r = 20
        tt = PNPTransistor(None, "", 1e-12, 25e-3, beta_f, beta_r)

        net = Network()
        v = net.addV("v", 6)
        re = net.addR("re", 10)
        rb = net.addR("rb", 20e3)
        rb2 = net.addR("rb2", 20e3)
        rc = net.addR("rc", 10)
        t1 = net.addComp("T1", tt)

        connect(v.p, re.p)
        connect(re.n, t1.C)
        connect(rb.n, net.ground)
        connect(rb.p, t1.B)
        connect(rb2.n, t1.B)
        connect(rb2.p, v.p)

        connect(t1.E, rc.p)
        connect(rc.n, v.n)
        connect(v.n, net.ground)
        ana = Analysis(net)
        res = ana.analyze(maxit=50)
        if isinstance(res, str):
            raise Exception(res)
        self.assertAlmostEqual(res.get_current(t1.E)/beta_r,res.get_current(t1.B),places=5)
        self.assertAlmostEqual(res.y_norm, 0)

class ResultDisplayTest(unittest.TestCase):

    def test_result_display(self):
        npntransistor = NPNTransistor(None, "", 1e-12, 25e-3, 100, 10)
        net = Network()
        v = net.addV("vc", 5)
        vb = net.addV("vb", Variable("vb", 2))
        rc = net.addR("rc", 100)
        rb = net.addR("rb", 1e3)
        t1 = net.addComp("T1", npntransistor)
        connect(v.p, rc.p)
        connect(rc.n, t1.C)
        connect(v.n, net.ground)
        connect(vb.p, rb.p)
        connect(rb.n, t1.B)
        connect(vb.n, net.ground)
        connect(t1.E, net.ground)
        ana = Analysis(net)
        res = ana.analyze(maxit=50)
        if isinstance(res, str):
            raise Exception(res)

        res.display()

class TransientTest(unittest.TestCase):

    def test1(self):

        v0 = 5.6
        r = 17e3
        capa = 312e-6
        timespan = 1
        net = Network()
        vc = net.addV("vc", v0)
        rc = net.addR("rc", r)
        ca = net.addCapa("ca", capa)

        connect(vc.p, rc.p)
        connect(rc.n, ca.p)
        connect(vc.n, net.ground)
        connect(ca.n, vc.n)

        ana = Analysis(net)
        # assumption capacitor is empty and will be loaded
        res = ana.transient(timespan,0.01, capa_voltages={"ca" : 0})

        import math
        ve = res[-1][1]["ca.p"]
        ve_expected = v0 *(1-math.exp(-timespan/(r*capa)))
        self.assertTrue(0.98 < ve/ve_expected <1.02)

    def test2(self):
        import math

        def waveform(t):
            return math.sin(t* 2* math.pi)

        net = Network()
        v0 = 10
        vc = net.addV("vc", v0, waveform= waveform)
        rc = net.addR("rc", 100)
        connect(vc.p, rc.p)
        connect(rc.n, vc.n)
        connect(vc.n, net.ground)
        ana = Analysis(net)
        res = ana.transient(1,0.005)

        for (t,v,c) in res:
            self.assertAlmostEqual(v["rc.p"], v0 * waveform(t))

    def test3(self):
        import math

        net = Network()
        v0 = 10
        ro = 1e1
        indo =  10
        curro = 1
        r = net.addR("rc", ro)
        ind = net.addInduc("ind", indo)
        connect(r.p, ind.p)
        connect(ind.n, r.n)
        connect(r.n, net.ground)
        ana = Analysis(net)
        res = ana.transient(1,0.0005, induc_currents={"ind": curro})
        #pp.pprint(res)
        a = res[0][2]["ind.p"]
        for (t,v,c) in res:
            curr = c["ind.p"]
            curr_expected = math.exp(-t*ro/indo)*curro
            self.assertTrue(0.98 < curr/curr_expected <1.02)


class TestInductor(unittest.TestCase):

    def test1(self):
        net = Network()
        v0 = 10
        vc = net.addV("vc", v0)
        r1 = net.addR("r1", 75)
        r2 = net.addR("r2", 25)
        ind = net.addInduc("ind", 100)
        connect(vc.p, r1.p)
        connect(r1.n, ind.p)
        connect(ind.n, r2.p)
        connect(r2.n, vc.n)
        connect(vc.n,net.ground)
        ana = Analysis(net)
        res = ana.analyze()
        self.assertAlmostEqual(res.get_voltage(ind.p), res.get_voltage(ind.n))
        self.assertAlmostEqual(res.get_current(ind.p), res.get_current(r1.p))
        self.assertAlmostEqual(res.get_current(ind.p), res.get_current(r2.p))
        self.assertAlmostEqual(res.get_voltage(ind.p), 2.5)

    def test2(self):
        net = Network()
        v0 = 10
        vc = net.addV("vc", v0)
        r1 = net.addR("r1", 75)
        r2 = net.addR("r2", 25)
        ind = net.addInduc("ind", 100)
        connect(vc.p, r1.p)
        connect(r1.n, ind.p)
        connect(ind.n, r2.p)
        connect(r2.n, vc.n)
        connect(vc.n,net.ground)
        ana = Analysis(net)
        res = ana.analyze(induc_currents={"ind": 0.09})
        #self.assertAlmostEqual(res.get_voltage(ind.p), res.get_voltage(ind.n))
        self.assertAlmostEqual(res.get_current(ind.p), res.get_current(r1.p))
        self.assertAlmostEqual(res.get_current(ind.p), res.get_current(r2.p))
        self.assertAlmostEqual(res.get_current(ind.p), 0.09)



if __name__ == '__main__':
    unittest.main()

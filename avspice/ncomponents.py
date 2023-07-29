""" the numerical components"""
import math
import numbers
from .util import explin, dexplin
from . import util


class NVoltage:
    """voltage source"""

    v: float

    def __init__(self, v: float):
        assert isinstance(v, (float, int))
        self.v = v

    def voltage(self, time: float) -> float:
        _ = time
        return self.v


class NSineVoltage:
    """sine voltage source"""

    def __init__(self, v: float, f: float):
        assert isinstance(v, (float, int))
        assert isinstance(f, (float, int))
        assert f > 0
        self.v = v
        self.f = f

    def voltage(self, time: float) -> float:
        return self.v * math.sin(2 * math.pi * self.f * time)


class NSawVoltage:
    """saw voltage source"""

    def __init__(self, v: float, f: float):
        self.v = v
        self.f = f

    def voltage(self, time: float) -> float:
        return self.v * util.saw_tooth(self.f, time)


class NPwmVoltage:
    """pwm voltage source"""

    def __init__(self, v: float, f: float, d: float):
        self.v = v
        self.f = f
        self.duty = d

    def voltage(self, time: float) -> float:
        frac, _ = math.modf(self.f * time)
        if frac > 1 - self.duty:
            return self.v
        else:
            return 0


class NPieceWiseLinearVoltage:
    """piecewise linear volatge source"""

    def __init__(self, vx: list[float], vy: list[float]):
        self.vx = vx
        self.vy = vy

    def voltage(self, time: float) -> float:
        return util.linear_interpolate(self.vx, self.vy, time)


class NPeriodicPieceWiseLinearVoltage:
    """piecewise linear voltage source"""

    def __init__(
        self,
        period: float,
        vx: list[float],
        vy: list[float],
        freq_mul: float,
        volt_mul: float,
    ):
        self.period = period
        ax = list(vx)
        ax.append(period)
        self.vx = ax
        ay = list(vy)
        ay.append(ay[0])
        self.vy = ay
        self.freq_mul = freq_mul
        self.volt_mul = volt_mul

    def voltage(self, time: float) -> float:
        time2 = time * self.freq_mul
        x = time2 % self.period
        return util.linear_interpolate(self.vx, self.vy, x) * self.volt_mul


class NSimpleVoltageControlledVoltageSource:
    """simple linear voltage controlled voltage source"""

    def __init__(self, factor: float):
        self.factor = factor

    def voltage(self, vin_diff: float) -> float:
        return self.factor * vin_diff

    def dvoltage(self, vin_diff: float) -> float:
        return self.factor


class NDiode:
    """solid state diode"""

    def __init__(
        self, Is: float, Nut: float, lcut_off: float = -40, rcut_off: float = 40
    ):
        self.Is = Is
        self.Nut = Nut

        self.lcut_off = lcut_off
        self.rcut_off = rcut_off

    def current(self, v: float) -> float:
        return self.Is * (explin(v / self.Nut, self.lcut_off, self.rcut_off) - 1)

    def diff_current(self, dv: float) -> float:
        return (
            self.Is
            * (1 / self.Nut)
            * dexplin(dv / self.Nut, self.lcut_off, self.rcut_off)
        )


class NZDiode:
    """solid state Z-diode"""

    def __init__(
        self,
        vcut: float,
        Is: float,
        Nut: float,
        IsZ: float,
        NutZ: float,
        lcut_off: float = -40,
        rcut_off: float = 40,
    ):
        assert isinstance(vcut, numbers.Number)
        self.vcut = vcut
        self.Is = Is
        self.Nut = Nut
        self.IsZ = IsZ
        self.NutZ = NutZ

        self.lcut_off = lcut_off
        self.rcut_off = rcut_off

    def current(self, v: float) -> float:
        pp = self.Is * (explin(v / self.Nut, self.lcut_off, self.rcut_off) - 1)
        nn = self.IsZ * (
            explin((-self.vcut) / self.NutZ, self.lcut_off, self.rcut_off)
            - explin((-self.vcut - v) / self.NutZ, self.lcut_off, self.rcut_off)
        )
        return nn + pp

    def diff_current(self, dv: float) -> float:
        dpp = (
            self.Is
            * (1 / self.Nut)
            * dexplin(dv / self.Nut, self.lcut_off, self.rcut_off)
        )
        dnn = (
            self.IsZ
            * (1 / self.NutZ)
            * dexplin((-self.vcut - dv) / self.NutZ, self.lcut_off, self.rcut_off)
        )

        return dnn + dpp


class NNPNTransistor:

    """NPN transistor"""

    def __init__(
        self,
        IS: float,
        VT: float,
        beta_F: float,
        beta_R: float,
        lcutoff: float,
        rcutoff: float,
    ):
        self.IS = IS
        self.VT = VT
        self.beta_F = beta_F
        self.beta_R = beta_R

        self.lcutoff = lcutoff
        self.rcutoff = rcutoff

    def t1(self, vbe: float, vbc: float) -> float:
        return explin(vbe / self.VT, self.lcutoff, self.rcutoff) - explin(
            vbc / self.VT, self.lcutoff, self.rcutoff
        )

    def d_t1_vbe(self, vbe: float) -> float:
        return dexplin(vbe / self.VT, self.lcutoff, self.rcutoff) / self.VT

    def d_t1_vbc(self, vbc: float) -> float:
        return -dexplin(vbc / self.VT, self.lcutoff, self.rcutoff) / self.VT

    def t2(self, vbc: float) -> float:
        return 1 / self.beta_R * (explin(vbc / self.VT, self.lcutoff, self.rcutoff) - 1)

    def d_t2_vbc(self, vbc: float) -> float:
        return (
            1
            / self.beta_R
            * dexplin(vbc / self.VT, self.lcutoff, self.rcutoff)
            / self.VT
        )

    def t3(self, vbe: float) -> float:
        return 1 / self.beta_F * (explin(vbe / self.VT, self.lcutoff, self.rcutoff) - 1)

    def d_t3_vbe(self, vbe: float) -> float:
        return (
            1
            / self.beta_F
            * dexplin(vbe / self.VT, self.lcutoff, self.rcutoff)
            / self.VT
        )

    def IC(self, vbe: float, vbc: float) -> float:
        return self.IS * (self.t1(vbe, vbc) - self.t2(vbc))

    def d_IC_vbe(self, vbe: float) -> float:
        return self.IS * self.d_t1_vbe(vbe)

    def d_IC_vbc(self, vbc: float) -> float:
        return self.IS * (self.d_t1_vbc(vbc) - self.d_t2_vbc(vbc))

    def IB(self, vbe: float, vbc: float) -> float:
        return self.IS * (self.t2(vbc) + self.t3(vbe))

    def d_IB_vbe(self, vbe: float) -> float:
        return self.IS * self.d_t3_vbe(vbe)

    def d_IB_vbc(self, vbc: float) -> float:
        return self.IS * self.d_t2_vbc(vbc)

    def IE(self, vbe: float, vbc: float) -> float:
        return self.IS * (self.t1(vbe, vbc) + self.t3(vbe))

    def d_IE_vbe(self, vbe: float) -> float:
        return self.IS * (self.d_t1_vbe(vbe) + self.d_t3_vbe(vbe))

    def d_IE_vbc(self, vbc: float) -> float:
        return self.IS * self.d_t1_vbc(vbc)


class NPNPTransistor:

    """PNP transistor"""

    def __init__(
        self,
        IS: float,
        VT: float,
        beta_F: float,
        beta_R: float,
        lcutoff: float = -40,
        rcutoff: float = 40,
    ):
        self.IS = IS
        self.VT = VT
        self.beta_F = beta_F
        self.beta_R = beta_R

        self.lcutoff = lcutoff
        self.rcutoff = rcutoff

    def t1(self, vbe: float, vbc: float) -> float:
        return explin(-vbe / self.VT, self.lcutoff, self.rcutoff) - explin(
            -vbc / self.VT, self.lcutoff, self.rcutoff
        )

    def d_t1_vbe(self, vbe: float) -> float:
        return -dexplin(-vbe / self.VT, self.lcutoff, self.rcutoff) / self.VT

    def d_t1_vbc(self, vbc: float) -> float:
        return dexplin(-vbc / self.VT, self.lcutoff, self.rcutoff) / self.VT

    def t2(self, vbc: float) -> float:
        return (
            1 / self.beta_R * (explin(-vbc / self.VT, self.lcutoff, self.rcutoff) - 1)
        )

    def d_t2_vbc(self, vbc: float) -> float:
        return (
            -1
            / self.beta_R
            * dexplin(-vbc / self.VT, self.lcutoff, self.rcutoff)
            / self.VT
        )

    def t3(self, vbe: float) -> float:
        return (
            1 / self.beta_F * (explin(-vbe / self.VT, self.lcutoff, self.rcutoff) - 1)
        )

    def d_t3_vbe(self, vbe: float) -> float:
        return (
            -1
            / self.beta_F
            * dexplin(-vbe / self.VT, self.lcutoff, self.rcutoff)
            / self.VT
        )

    # ---
    def IC(self, vbe: float, vbc: float) -> float:
        return self.IS * (self.t2(vbc) - self.t1(vbe, vbc))

    def d_IC_vbe(self, vbe: float) -> float:
        return -self.IS * self.d_t1_vbe(vbe)

    def d_IC_vbc(self, vbc: float) -> float:
        return self.IS * (self.d_t2_vbc(vbc) - self.d_t1_vbc(vbc))

    def IB(self, vbe: float, vbc: float) -> float:
        return -self.IS * (self.t2(vbc) + self.t3(vbe))

    def d_IB_vbe(self, vbe: float) -> float:
        return -self.IS * self.d_t3_vbe(vbe)

    def d_IB_vbc(self, vbc: float) -> float:
        return -self.IS * self.d_t2_vbc(vbc)

    def IE(self, vbe: float, vbc: float) -> float:
        return self.IS * (self.t1(vbe, vbc) + self.t3(vbe))

    def d_IE_vbe(self, vbe: float) -> float:
        return self.IS * (self.d_t1_vbe(vbe) + self.d_t3_vbe(vbe))

    def d_IE_vbc(self, vbc: float) -> float:
        return self.IS * self.d_t1_vbc(vbc)


class NFET:
    """FET"""

    def __init__(self, vth: float):
        self.vth = vth  # threshold voltage
        self.id0 = 1e-5
        self.vt = 25e-3  # temperature voltage
        self.n = 1
        self.knp = 1.0 / 5.0

    # S is ground
    def IS(self, vgs: float, vds: float) -> float:
        if vds < 0:
            # we should use the body diode!
            return 0
        if vgs < self.vth:
            return self.id0 * math.exp((vgs - self.vth) / (self.vt * self.n))

        if vds < vgs - self.vth:
            return self.knp * ((vgs - self.vth) * vds - vds * vds / 2) + self.id0

        # vds >= vgs - self.vth:
        return self.knp * (vgs - self.vth) * (vgs - self.vth) / 2 + self.id0

    def d_IS_vgs(self, vgs: float, vds: float) -> float:
        if vds < 0:
            return 0
        if vgs < self.vth:
            return (
                self.id0
                * math.exp((vgs - self.vth) / (self.vt * self.n))
                / (self.vt * self.n)
            )
        if vgs - self.vth > vds:
            return self.knp * (vgs - self.vth)

        return self.knp * 2 * (vgs - self.vth)

    def d_IS_vds(self, vgs: float, vds: float) -> float:
        if vds < 0:
            return 0
        if vgs < self.vth:
            return 0
        if vds < vgs - self.vth:
            return self.knp * (vgs - self.vth - vds)

        # vds >= vgs - self.vth
        return 0


class NJFETn:
    """n channel JFET

    the wikipedia entry is not that great
    https://www.mathworks.com/help/sps/ref/nchanneljfet.html is OK


    In general V_D > V_S, i.e. D is at +, S is at -

    component is naturally on, depletion mode

    V_G must be less than V_S, this will lower the current from D to S.
    If V_G < V_th than no current flows. V_th: threshold voltage


    if V_G is larger than V_S or V_D than a diode current will flow to respctive gate.

    if V_S > V_D the component works as well but a little but different, since
    the component is not symmetric.


    Assumption V_DS >= 0


    off:
    V_GS <  V_th:  I_D = 0

    independent of V_DS

    linear:
    0 < V_DS < V_GS – V_th
    I_D = beta * V_DS(2(V_GS – V_th) – V_DS)(1 +  lambda V_DS)

    saturated:
    V_GS - V_th < V_DS
    I_D = beta ( V_GS -V_th) ^2 * (1 + lambda V_DS)

    beta transconductnace parameter
    lambda channel length parameter



    if V_D < V_S, i.e. V_DS <0 then its the same formula:
    I_D(V_GS, V_DS)
     = - I_D(V_GD, V_SD)
     =  -I(V_GS - V_DS, -V_DS)



    """

    def __init__(self, v_th: float, beta: float, lambda_: float):
        self.v_th = v_th  # threshold voltage
        self.beta = beta
        self.lambda_ = lambda_

    # S is ground

    def IS(self, vgs: float, vds: float) -> float:
        if vds >= 0:
            if vgs < self.v_th:
                return 0
            if vds <= vgs - self.v_th:
                return (
                    self.beta
                    * vds
                    * (2 * (vgs - self.v_th) - vds)
                    * (1 + self.lambda_ * vds)
                )
            else:
                return self.beta * (vgs - self.v_th) ** 2 * (1 + self.lambda_ * vds)
        else:
            return -self.IS(vgs - vds, -vds)

    def d_IS_vgs(self, vgs: float, vds: float) -> float:
        if vds >= 0:
            if vgs < self.v_th:
                return 0
            if vds <= vgs - self.v_th:
                return self.beta * vds * 2 * (1 + self.lambda_ * vds)
            else:
                return self.beta * 2 * (vgs - self.v_th) * (1 + self.lambda_ * vds)
        else:
            return -self.d_IS_vgs(vgs - vds, -vds)

    def d_IS_vds(self, vgs: float, vds: float) -> float:
        if vds >= 0:
            if vgs < self.v_th:
                return 0
            if vds <= vgs - self.v_th:
                return self.beta * (
                    2 * (vgs - self.v_th)
                    + (2 * self.lambda_ * (vgs - self.v_th) - 1) * 2 * vds
                    - self.lambda_ * 3 * vds**2
                )

            else:
                return self.beta * (vgs - self.v_th) ** 2 * self.lambda_
        else:
            return self.d_IS_vgs(vgs - vds, -vds) + self.d_IS_vds(vgs - vds, -vds)

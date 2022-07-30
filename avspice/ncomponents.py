""" the numerical components"""
import math
import numbers
from .util import explin, dexplin, smooth_step, dsmooth_step
from . import util



class NVoltage:
    """voltage source"""

    def __init__(self, v):
        assert isinstance(v, (float,int))
        self.v = v

    def voltage(self, time):
        return self.v

class NSineVoltage:
    """sine voltage source"""
    def __init__(self, v, f):
        assert isinstance(v, (float,int))
        assert isinstance(f, (float,int))
        assert f>0
        self.v = v
        self.f = f


    def voltage(self, time):
        return self.v * math.sin(2 * math.pi * self.f * time)

class NSawVoltage:
    """savw voltage source"""
    def __init__(self, v, f):
        self.v = v
        self.f = f

    def voltage(self, time):
        return self.v * util.saw_tooth(self.f, time)

class NPieceWiseLinearVoltage:
    """piecewise linear volatge source"""
    def __init__(self, vx, vy):
        self.vx = vx
        self.vy = vy

    def voltage(self, time):
        return util.linear_interpolate(self.vx, self.vy, time)

class NDiode:
    """solid state diode"""
    def __init__(self, Is, Nut, lcut_off = -40, rcut_off=40):
        self.Is = Is
        self.Nut = Nut


        self.lcut_off = lcut_off
        self.rcut_off = rcut_off

    def current(self, v):
        return self.Is * (explin(v/self.Nut, self.lcut_off, self.rcut_off)-1)

    def diff_current(self, dv):
        return self.Is * (1/self.Nut) * dexplin(dv/self.Nut, self.lcut_off, self.rcut_off)

class NZDiode:
    """solid state Z-diode"""
    def __init__(self, vcut, Is, Nut, IsZ, NutZ, lcut_off = -40, rcut_off=40):
        assert isinstance(vcut, numbers.Number)
        self.vcut = vcut
        self.Is = Is
        self.Nut = Nut
        self.IsZ = IsZ
        self.NutZ = NutZ

        self.lcut_off = lcut_off
        self.rcut_off = rcut_off

    def current(self, v):
        pp = self.Is * (explin(v/self.Nut, self.lcut_off, self.rcut_off)-1)
        nn = self.IsZ * ( #1-
                         explin((-self.vcut)/self.NutZ, self.lcut_off, self.rcut_off)-
                         explin((-self.vcut-v)/self.NutZ, self.lcut_off, self.rcut_off))
        a = smooth_step(-self.vcut/2,0, v)
        return nn + a * (pp-nn)

    def diff_current(self, dv):
        pp = self.Is * (explin(dv/self.Nut, self.lcut_off, self.rcut_off)-1)
        nn = self.IsZ * (explin((-self.vcut)/self.NutZ, self.lcut_off, self.rcut_off)
                         -explin((-self.vcut-dv)/self.NutZ, self.lcut_off, self.rcut_off))

        dpp =   self.Is * (1/self.Nut) * dexplin(dv/self.Nut, self.lcut_off, self.rcut_off)
        dnn =   (self.IsZ * (1/self.NutZ)
                 * dexplin((-self.vcut - dv)/self.NutZ, self.lcut_off, self.rcut_off))
        a = smooth_step(-self.vcut/2,0, dv)
        da = dsmooth_step(-self.vcut/2,0, dv)

        return dnn + a * (dpp - dnn)  + da * (pp - nn)

class NNPNTransistor:

    """NPN transistor"""

    def __init__(self, IS:float, VT:float, beta_F:float, beta_R:float,
                 lcutoff:float,
                 rcutoff:float):
        self.IS = IS
        self.VT = VT
        self.beta_F = beta_F
        self.beta_R = beta_R

        self.lcutoff = lcutoff
        self.rcutoff = rcutoff

    def t1(self, vbe, vbc):
        return (explin(vbe/self.VT, self.lcutoff, self.rcutoff)
                - explin(vbc/self.VT, self.lcutoff, self.rcutoff))

    def d_t1_vbe(self, vbe):
        return dexplin(vbe/self.VT, self.lcutoff, self.rcutoff) / self.VT

    def d_t1_vbc(self, vbc):
        return -dexplin(vbc/self.VT, self.lcutoff, self.rcutoff) / self.VT

    def t2(self, vbc):
        return 1/self.beta_R *(explin(vbc/self.VT, self.lcutoff, self.rcutoff)-1)

    def d_t2_vbc(self, vbc):
        return 1/self.beta_R * dexplin(vbc/self.VT, self.lcutoff, self.rcutoff) /self.VT

    def t3(self, vbe):
        return 1/self.beta_F *(explin(vbe/self.VT, self.lcutoff, self.rcutoff)-1)

    def d_t3_vbe(self, vbe):
        return 1/self.beta_F * dexplin(vbe/self.VT, self.lcutoff, self.rcutoff) / self.VT

    def IC(self, vbe, vbc):
        return self.IS*(self.t1(vbe, vbc) - self.t2(vbc))
    def d_IC_vbe(self, vbe):
        return self.IS * self.d_t1_vbe(vbe)

    def d_IC_vbc(self, vbc):
        return self.IS * (self.d_t1_vbc(vbc) - self.d_t2_vbc(vbc))

    def IB(self, vbe, vbc):
        return self.IS * (self.t2(vbc) + self.t3(vbe))

    def d_IB_vbe(self, vbe):
        return self.IS * self.d_t3_vbe(vbe)

    def d_IB_vbc(self, vbc):
        return self.IS * self.d_t2_vbc(vbc)


    def IE(self, vbe, vbc):
        return self.IS * (self.t1(vbe, vbc) + self.t3(vbe))

    def d_IE_vbe(self, vbe):
        return self.IS * (self.d_t1_vbe(vbe) + self.d_t3_vbe(vbe))

    def d_IE_vbc(self, vbc):
        return self.IS * self.d_t1_vbc(vbc)

class NPNPTransistor:

    """PNP transistor"""

    def __init__(self, IS:float, VT:float, beta_F:float, beta_R:float,
                 lcutoff:float = -40,
                 rcutoff:float = 40):

        self.IS = IS
        self.VT = VT
        self.beta_F = beta_F
        self.beta_R = beta_R

        self.lcutoff = lcutoff
        self.rcutoff = rcutoff

    def t1(self, vbe, vbc):
        return (explin(-vbe/self.VT, self.lcutoff, self.rcutoff)
                - explin(-vbc/self.VT, self.lcutoff, self.rcutoff))

    def d_t1_vbe(self, vbe):
        return -dexplin(-vbe/self.VT, self.lcutoff, self.rcutoff) / self.VT

    def d_t1_vbc(self, vbc):
        return dexplin(-vbc/self.VT, self.lcutoff, self.rcutoff) / self.VT

    def t2(self, vbc):
        return 1/self.beta_R *(explin(-vbc/self.VT, self.lcutoff, self.rcutoff)-1)

    def d_t2_vbc(self, vbc):
        return -1/self.beta_R * dexplin(-vbc/self.VT, self.lcutoff, self.rcutoff) /self.VT

    def t3(self, vbe):
        return 1/self.beta_F *(explin(-vbe/self.VT, self.lcutoff, self.rcutoff)-1)

    def d_t3_vbe(self, vbe):
        return -1/self.beta_F * dexplin(-vbe/self.VT, self.lcutoff, self.rcutoff) / self.VT

    #---
    def IC(self, vbe, vbc):
        return self.IS*(self.t2(vbc) - self.t1(vbe, vbc))

    def d_IC_vbe(self, vbe):
        return -self.IS * self.d_t1_vbe(vbe)

    def d_IC_vbc(self, vbc):
        return self.IS * (self.d_t2_vbc(vbc) - self.d_t1_vbc(vbc))


    def IB(self, vbe, vbc):
        return -self.IS * (self.t2(vbc) + self.t3(vbe))

    def d_IB_vbe(self, vbe):
        return -self.IS * self.d_t3_vbe(vbe)

    def d_IB_vbc(self, vbc):
        return -self.IS * self.d_t2_vbc(vbc)


    def IE(self, vbe, vbc):
        return self.IS * (self.t1(vbe, vbc) + self.t3(vbe))

    def d_IE_vbe(self, vbe):
        return self.IS * (self.d_t1_vbe(vbe) + self.d_t3_vbe(vbe))

    def d_IE_vbc(self, vbc):
        return self.IS * self.d_t1_vbc(vbc)

""" simple routines for experimenting with nodal analysis """

import math
import pprint as pp
import numbers
import numpy as np
from solving import solve as solve

def explin(x: float, lcutoff: float, rcutoff:float):
    assert lcutoff  <= rcutoff, "cutoffs wrong"
    if lcutoff <=  x <= rcutoff:
        return math.exp(x)
    if x > rcutoff:
        return math.exp(rcutoff) +  (x-rcutoff) * math.exp(rcutoff)
    if x < lcutoff:
        return math.exp(lcutoff) +  (x-lcutoff) * math.exp(lcutoff)


def dexplin(x:float, lcutoff:float, rcutoff:float):
    assert lcutoff  <= rcutoff, "cutoffs wrong"
    if lcutoff <=  x <= rcutoff:
        return math.exp(x)
    if x > rcutoff:
        return math.exp(rcutoff)
    if x < lcutoff:
        return  math.exp(lcutoff)

class Variable:
    """a variable"""
    def __init__(self, name, default=None):
        self.name = name
        self.default = default

    def __repr__(self):
        return "<Variable {0}, {1}>".format(self.name, self.default)

class Component:
    """Component in a electrical network, e.g. resistor, current source, node"""
    def __init__(self, parent, name):
        self.name = name
        self.parent = parent

    def ports(self):
        """return the ports of this component"""
        raise NotImplementedError("ports method not implemented")

    def get_port(self, name):
        for p in self.ports():
            if p.name == name:
                return p
        raise Exception("Component {0} does not have port {1}".format(self.name, name))

    def get_val(self, var_or_num, variables):
        if isinstance(var_or_num, numbers.Number):
            return var_or_num
        if isinstance(var_or_num, Variable):
            val = variables.get(var_or_num.name, None)
            if val is None:
                if var_or_num.default is None:
                    raise Exception("did not find value for var {0} in {1}"
                                    .format(var_or_num, self.name))
                return var_or_num.default
            return val
        raise Exception("bug")

class Port:
    """ components are connected via their ports"""

    def __init__(self, component:Component, name:str):
        self.component = component
        self.name = name

    def __repr__(self):
        return "<Port {0}.{1}>".format(self.component.name, self.name)

    def pname(self):
        """printable name of the port"""
        return self.component.name + "." + self.name


class Node2(Component):
    """a component with just two ports"""

    def __init__(self, parent: 'Network', name:str):
        super().__init__(parent, name)
        self.p = Port(self, "p")
        self.n = Port(self, "n")

    def ports(self):
        return [self.p, self.n]

    def get_current(self, variables, vd):
        raise NotImplementedError("get_current method not implemented")

class Node(Component):
    """a node, just a port in the network"""
    def __init__(self, parent, name):
        super().__init__(parent, name)
        self.port = Port(self, "port")

    def ports(self):
        return [self.port]

    def __repr__(self):
        return "<Node {0}>".format(self.name)


class Resistor(Node2):
    """resistor"""
    def __init__(self, parent, name, ohm):
        super().__init__(parent, name)
        self._ohm = ohm

    def get_ohm(self, variables):
        return self.get_val(self._ohm, variables)

    def __repr__(self):
        return "<Resistor {0}>".format(self.name)

    def get_current(self,variables, vd):
        return vd / self.get_ohm(variables)


class Current(Node2):
    """current source"""
    def __init__(self, parent, name, amp):
        super().__init__(parent, name)
        self.amp = amp

    def __repr__(self):
        return "<Current {0}>".format(self.name)

    def get_amp(self, variables):
        return self.get_val(self.amp, variables)

    def get_current(self, variables, vd):
        return self.get_val(self.amp, variables)

class Voltage(Node2):
    """voltage source"""
    def __init__(self, parent, name:str, volts:float):
        super().__init__(parent, name)
        assert isinstance(volts, (numbers.Number, Variable)), "volts must be a variable or a number"
        self.volts = volts

    def voltage(self, variables):
        return self.get_val(self.volts, variables)

    def __repr__(self):
        return "<Voltage {0}>".format(self.name)

    def get_current(self, variables, vd):
        raise NotImplementedError("get_current for voltage source not implemented")

class Diode(Node2):
    """solid state diode"""
    def __init__(self, parent, name, Is, Nut, lcut_off = -40, rcut_off=40):
        super().__init__(parent, name)
        self.Is = Is
        self.Nut = Nut

        self.lcut_off = lcut_off
        self.rcut_off = rcut_off

    def current(self, v):
        return self.Is * (explin(v/self.Nut, self.lcut_off, self.rcut_off)-1)

    def diff_conductance(self, v):
        if v<0:
            return 0
        return self.Is * 1/self.Nut * dexplin(v/self.Nut, self.cut_off)

    def get_current(self, variables, vd):
        return self.Is * (explin(vd/self.Nut, self.lcut_off, self.rcut_off)-1)

    def diff_curr(self, dv):
        return self.Is * (1/self.Nut) * dexplin(dv/self.Nut, self.lcut_off, self.rcut_off)

    def __repr__(self):
        return "<Diode {0}>".format(self.name)

class Capacitor(Node2):
    """ a capacitor"""

    def __init__(self, parent, name, capa):
        super().__init__(parent, name)
        self.capa = capa

    def get_current(self, variables, vd):
        raise NotImplementedError("get_current for capacitor not implemented")


class NPNTransistor(Component):
    """npn transistor

    model Ebers Moll, according to wikipedia

    IS: the reverse saturation current (on the order of 10−15 to 10−12 amperes)
    VT: the thermal voltage (approximately 26 mV at 300 K ≈ room temperature).
    beta_F: the forward common emitter current gain (20 to 500)
    beta_R: the reverse common emitter current gain (0 to 20)

    t1 = exp(V_BE/VT) - exp(V_BC/VT)
    t2 = 1/beta_R (exp(V_BC/VT) -1)
    t3 = 1/beta_F (exp(V_BE/VT) -1)

    I_C = IS(t1 - t2)
    I_B = IS(t2 + t3)
    I_E = IS(t1 + t3)

    My interpretation:
    t3 is the current from base to emitter
    t2 is the current from base to collector
    t1 is controlled current

    """

    def __init__(self, parent: 'Network', name:str, IS:float, VT:float, beta_F:float, beta_R:float,
                 cutoff:float =40):
        super().__init__(parent, name)
        self.IS = IS
        self.VT = VT
        self.beta_F = beta_F
        self.beta_R = beta_R
        self.B = Port(self,"B")
        self.C = Port(self,"C")
        self.E = Port(self,"E")

        self.lcutoff = -cutoff
        self.rcutoff = cutoff


    def ports(self):
        return [self.B, self.C, self.E]

    def t1(self, vbe, vbc):
        return explin(vbe/self.VT, self.lcutoff, self.rcutoff) - explin(vbc/self.VT, self.lcutoff, self.rcutoff)

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

    def get_current3(self, variables, vbe, vbc):
        return (self.IB(vbe, vbc),  -self.IE(vbe, vbc), self.IC(vbe, vbc))


class PNPTransistor(Component):


    """


    t1 = exp(-V_BE/VT) - exp(-V_BC/VT)
    t2 = 1/beta_R (exp(-V_BC/VT) -1)
    t3 = 1/beta_F (exp(-V_BE/VT) -1)

    Emitter is positive

    I_C = IS(-t1 + t2) = IS(t2 - t1)
    I_B = IS(-t2 -t3) = -IS(t2 + t3)
    I_E = IS(t1 + t3) =  IS(t1 + t3)

    My interpretation:
    t3 is the current from emitter to base
    t2 is the current from collector to base
    t1 is controlled current, emitter to collector



    PNP!
    """

    def __init__(self, parent: 'Network', name:str, IS:float, VT:float, beta_F:float, beta_R:float,
                 cutoff:float =40):
        super().__init__(parent, name)
        self.IS = IS
        self.VT = VT
        self.beta_F = beta_F
        self.beta_R = beta_R
        self.B = Port(self,"B")
        self.C = Port(self,"C")
        self.E = Port(self,"E")

        self.lcutoff = -cutoff
        self.rcutoff = cutoff

    def ports(self):
        return [self.B, self.C, self.E]

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

    def get_current3(self, variables, vbe, vbc):
        return (self.IB(vbe, vbc),  self.IE(vbe, vbc), self.IC(vbe, vbc))

class Network:
    """ this class describes the toplogy of an electrical network
        It only contains the topology"""

    def __init__(self):
        self.connections = []
        self.ground = Node(self, "ground")
        self.components = { self.ground.name : self.ground}

    def addR(self, name:str, ohm:float):
        """ add a resistor """
        if name in self.components:
            raise Exception("Name {0} already exists".format(name))
        r = Resistor(self, name , ohm)
        self.components[name] = r
        return r

    def addC(self, name, amp):
        """add a curent source"""
        if name in self.components:
            raise Exception("Name {0} already exists".format(name))
        c = Current(self, name, amp)
        self.components[name] = c
        return c

    def addV(self, name, volts):
        """add a voltage source"""
        if name in self.components:
            raise Exception("Name {0} already exists".format(name))
        v = Voltage(self, name, volts)
        self.components[name] = v
        return v

    def addN(self,name):
        """add a node"""
        if name in self.components:
            raise Exception("Name {0} already exists".format(name))
        node = Node(self, name)
        self.components[name] = node
        return node

    def addD(self, name, Is, Nut):
        if name in self.components:
            raise Exception("Name {0} already exists".format(name))
        d = Diode(self, name, Is, Nut)
        self.components[name] = d
        return d

    def addComp(self, name, comp):
        if name in self.components:
            raise Exception("Name {0} already exists".format(name))
        if isinstance(comp, Diode):
            d = Diode(self, name, comp.Is, comp.Nut, lcut_off = comp.lcut_off, rcut_off = comp.rcut_off)
            self.components[name] = d
            return d
        if isinstance(comp, NPNTransistor):
            t = NPNTransistor(self, name, comp.IS, comp.VT, comp.beta_F, comp.beta_R)
            self.components[name] = t
            return t
        if isinstance(comp, PNPTransistor):
            t = PNPTransistor(self, name, comp.IS, comp.VT, comp.beta_F, comp.beta_R)
            self.components[name] = t
            return t
        raise Exception("addComp not supported for {0}".format(comp))

    def addCapa(self, name, capa):
        if name in self.components:
            raise Exception("Name {0} already exists".format(name))
        c = Capacitor(self, name, capa)
        self.components[name] = c
        return c

    def addConnection(self, p1, p2):
        """connect two ports"""
        if isinstance(p2, Node):
            p2 = p2.port
        if isinstance(p1, Node):
            p1 = p1.port

        c1 = p1.component
        c2 = p2.component
        if c1.parent != self or c2.parent != self:
            raise Exception("wrong network for addConnection")

        self.connections.append((p1, p2))

    def get_object(self, name):
        l = name.split(".")
        name = l[0]
        if not name in self.components:
            raise Exception("unknown component: {0}".format(name))
        c = self.components[name]
        if len(l) == 1:
            return c
        if len(l) == 2:
            return c.get_port(l[1])
        raise Exception("too many components in name {0}".format(name))

    def get_capacitors(self):
        res = []
        for c in self.components.values():
            if isinstance(c,Capacitor):
                res.append(c)
        return res




def connect(p1,p2):
    """connect two ports or nodes"""
    if isinstance(p1, Node):
        compo = p1
    else:
        compo = p1.component
    compo.parent.addConnection(p1,p2)

class XNode():
    """node class for analysis"""
    def __init__(self, name, ports):
        self.ports = ports
        self.name = name

    def __repr__(self):
        return "<XNode {0}: {1}>".format(self.name, repr(self.ports))

def mk_xnode(nodes):
    """create an XNode for the given ports and derive a name"""

    def key(x):
        if isinstance(x.component, Node):
            return "a" +  x.component.name
        return "b" + x.component.name

    l = list(nodes)
    l.sort(key=key)
    if isinstance(l[0].component, Node):
        s = None
        for x in l:
            if not isinstance(x.component, Node):
                break
            if s:
                s= s+ "/" + x.component.name
            else:
                s= x.component.name
    else:
        s = None
        for p in l:

            if s:
                s= s+ "/" + p.pname()
            else:
                s= p.pname()
    return XNode(s, nodes)


def compute_nodes(nw):
    """ compute the nodes for a network, i.e. connected ports"""

    allports = set()
    for comp in nw.components.values():
        for port in comp.ports():
            allports.add(port)

    ## adjancency
    adj:str = dict()
    adj[nw.ground.port] = []

    def add(p1, p2):
        if p1 in adj:
            adj[p1].append(p2)
        else:
            adj[p1] = [p2]

    for (x, y) in nw.connections:
        add(x, y)
        add(y, x)

    done = set()
    nodes = []
    todo = [nw.ground.port]
    while todo:
        port = todo.pop()
        if port in done:
            continue

        stack = [port]
        node = set()
        done.add(port)
        allports.remove(port)
        while stack:
            x = stack.pop()
            for p in x.component.ports():
                if not p in done:
                    todo.append(p)
            node.add(x)
            if not x in adj:
                raise Exception("{0} is not connected".format(x))
            for p1 in adj[x]:
                if p1 in done:
                    continue
                stack.append(p1)
                allports.remove(p1)
                done.add(p1)
        # per algorithm ground is the first node
        nodes.append(mk_xnode(node))
    if allports:
        pp.pprint(allports)
        raise Exception("some ports are not connected to ground: {0}".format(allports))
    return nodes

class Result:
    """result of an analysis run"""
    def __init__(self, network, analysis, iterations, solution_vec, variables, y, y_norm, mat_cond):
        self.analysis = analysis
        self.network = network
        self.solution_vec = solution_vec
        self.voltages = dict()
        self.currents = dict()
        self.iterations = iterations
        self.mat_cond = mat_cond
        self.y = y
        self.y_norm = y_norm
        for c in network.components.values():
            if isinstance(c, Capacitor):
                k = self.analysis.capa_index(c)
                cu = solution_vec[k]
                self.currents[c.p] = -cu
                self.currents[c.n] = cu
            elif isinstance(c, Voltage):
                k = self.analysis.voltage_index(c)
                cu = solution_vec[k]
                self.currents[c.p] = -cu
                self.currents[c.n] = cu
            elif isinstance(c, Node2):
                kp = self.analysis.port_index(c.p)
                kn = self.analysis.port_index(c.n)
                vd = self.solution_vec[kp] - self.solution_vec[kn]
                i = c.get_current(variables, vd)
                self.currents[c.p] = i
                self.currents[c.n] = -i
            elif isinstance(c, NPNTransistor) or isinstance(c, PNPTransistor):
                vb = self.get_voltage(c.B)
                ve = self.get_voltage(c.E)
                vc = self.get_voltage(c.C)
                (ib, ie, ic) = c.get_current3(variables, vb - ve, vb- vc)
                self.currents[c.B] = ib
                self.currents[c.E] = ie
                self.currents[c.C] = ic
            elif  c == self.network.ground:
                pass
            else:
                raise Exception("unknown component:{0} ".format(c))

            for port in c.ports():
                k = self.analysis.port_index(port)
                self.voltages[port] = solution_vec[k]

    def get_voltages(self):
        res = dict()
        for (p,v) in self.voltages.items():
            res[p.pname()] = v
        return res

    def get_currents(self):
        res = dict()
        for (p,c) in self.currents.items():
            res[p.pname()] = c
        return res

    def __repr__(self):
        return repr({"voltages": self.voltages, "currents": self.currents})

    def get_voltage(self, name_or_object):
        port = self._port(name_or_object)
        k = self.analysis.port_index(port)
        return self.solution_vec[k]

    def get_avoltage(self, name_or_object):
        c = None
        if isinstance(name_or_object, str):
            c = self.network.get_object(name_or_object)
        if not isinstance(c, Node2):
            raise Exception("{0} is not a Node2 or name for one".format(name_or_object))
        return self.voltages[c.p] - self.voltages[c.n]

    def _port(self, name_or_comp):
        c = None
        if isinstance(name_or_comp, str):
            c = self.network.get_object(name_or_comp)
        else:
            c = name_or_comp
        if not isinstance(c, Port):
            raise Exception("not a port or node or name thereof: {0}".format(name_or_comp))
        return c

    def get_current(self, name_or_comp):
        c = self._port(name_or_comp)
        return self.currents[c]

    def has_current(self, name_or_comp):
        c = self._port(name_or_comp)
        return c in self.currents

    def display(self):
        ports = []
        compos = []
        for comp in self.network.components.values():
            compos.append(comp.name)
            for port in comp.ports():
                ports.append(port.pname())
        ports.sort()
        print("--- Voltages ---")
        for port in ports:
            print(port + " " + str(self.get_voltage(port)))
        compos.sort()
        print(" --- currents ---")
        for cname in compos:
            comp = self.network.get_object(cname)
            if not isinstance(comp, Node):
                for port in comp.ports():
                    if self.has_current(port):
                        print(port.pname() + " " + str(self.get_current(port)))

class Analysis:
    """captures all data for analysis"""

    def __init__(self, netw):
        self.netw = netw
        self._port_to_node = dict()
        self.node_list = []
        self.voltage_list = []
        self.capa_list = []
        self.ground = None
        self.mat = None
        self.r = None
        self.solution_vec = None
        self.node_list = compute_nodes(self.netw)
        self.ground = self.node_list[0]
        self.energy_factor = 1

        for node in self.node_list:
            for port in node.ports:
                self._port_to_node[port] = node

        for comp in self.netw.components.values():
            if isinstance(comp, Voltage):
                self.voltage_list.append(comp)
            if isinstance(comp, Capacitor):
                self.capa_list.append(comp)



    def node_ports(self, node):
        return node.ports

    def port_node(self, port):
        return self._port_to_node[port]

    def node_index(self, node):
        return self.node_list.index(node)

    def port_index(self, port):
        return self.node_index(self.port_node(port))

    def voltage_index(self, voltage):
        return self.voltage_list.index(voltage) + len(self.node_list)

    def capa_index(self, capa):
        return self.capa_list.index(capa) + len(self.node_list) + len(self.voltage_list)

    def voltage(self, solution_vec, port):
        k = self.port_index(port)
        return solution_vec[k]

    def process_diode_y(self, comp, sol, y, variables):
        kp = self.port_index(comp.p)
        kn = self.port_index(comp.n)
        dv = sol[kp] - sol[kn]
        curr = comp.current(dv)
        y[kp] -= curr
        y[kn] += curr

    def process_diode_D(self, comp, sol, D, variables):
        kp = self.port_index(comp.p)
        kn = self.port_index(comp.n)
        dv = sol[kp] - sol[kn]
        diff = comp.diff_curr(dv)
        D[kp][kp] -= diff
        D[kp][kn] += diff
        D[kn][kp] += diff
        D[kn][kn] -= diff


    def process_npn_transistor_y(self, tra, sol, y, variables):
        kB = self.port_index(tra.B)
        kE = self.port_index(tra.E)
        kC = self.port_index(tra.C)

        vbe0 = self.voltage(sol, tra.B) - self.voltage(sol, tra.E)
        vbc0 = (self.voltage(sol, tra.B) - self.voltage(sol, tra.C))

        ie = tra.IE(vbe0, vbc0)
        ib = tra.IB(vbe0, vbc0)
        ic = tra.IC(vbe0, vbc0)

        y[kE]+= ie
        y[kB]-= ib
        y[kC]-= ic

    def process_npn_transistor_D(self, tra, sol, D, variables):
        kB = self.port_index(tra.B)
        kE = self.port_index(tra.E)
        kC = self.port_index(tra.C)

        vbe0 = self.voltage(sol, tra.B) - self.voltage(sol, tra.E)
        vbc0 = (self.voltage(sol, tra.B) - self.voltage(sol, tra.C))

        D[kB][kB] += -tra.d_IB_vbe(vbe0)
        D[kB][kB] += -tra.d_IB_vbc(vbc0)
        D[kB][kE] += tra.d_IB_vbe(vbe0)
        D[kB][kC] += tra.d_IB_vbc(vbc0)

        D[kC][kB] -= tra.d_IC_vbe(vbe0)
        D[kC][kB] -= tra.d_IC_vbc(vbc0)
        D[kC][kE] += tra.d_IC_vbe(vbe0)
        D[kC][kC] += tra.d_IC_vbc(vbc0)

        D[kE][kB] += tra.d_IE_vbe(vbe0)
        D[kE][kB] += tra.d_IE_vbc(vbc0)
        D[kE][kE] -= tra.d_IE_vbe(vbe0)
        D[kE][kC] -= tra.d_IE_vbc(vbc0)

    def process_pnp_transistor_y(self, tra, sol, y, variables):
        kB = self.port_index(tra.B)
        kE = self.port_index(tra.E)
        kC = self.port_index(tra.C)

        vbe0 = self.voltage(sol, tra.B) - self.voltage(sol, tra.E)
        vbc0 = (self.voltage(sol, tra.B) - self.voltage(sol, tra.C))

        ie = tra.IE(vbe0, vbc0)
        ib = tra.IB(vbe0, vbc0)
        ic = tra.IC(vbe0, vbc0)

        y[kE]-= ie
        y[kB]-= ib
        y[kC]-= ic

    def process_pnp_transistor_D(self, tra, sol, D, variables):
        kB = self.port_index(tra.B)
        kE = self.port_index(tra.E)
        kC = self.port_index(tra.C)

        vbe0 = self.voltage(sol, tra.B) - self.voltage(sol, tra.E)
        vbc0 = (self.voltage(sol, tra.B) - self.voltage(sol, tra.C))

        D[kB][kB] += -tra.d_IB_vbe(vbe0)
        D[kB][kB] += -tra.d_IB_vbc(vbc0)
        D[kB][kE] += tra.d_IB_vbe(vbe0)
        D[kB][kC] += tra.d_IB_vbc(vbc0)

        D[kC][kB] -= tra.d_IC_vbe(vbe0)
        D[kC][kB] -= tra.d_IC_vbc(vbc0)
        D[kC][kE] += tra.d_IC_vbe(vbe0)
        D[kC][kC] += tra.d_IC_vbc(vbc0)

        D[kE][kB] -= tra.d_IE_vbe(vbe0)
        D[kE][kB] -= tra.d_IE_vbc(vbc0)
        D[kE][kE] += tra.d_IE_vbe(vbe0)
        D[kE][kC] += tra.d_IE_vbc(vbc0)


    def process_voltage_y(self, vol: Voltage, sol, y, variables):
        k = self.voltage_index(vol)
        kp = self.port_index(vol.p)
        kn = self.port_index(vol.n)
        y[kp] += sol[k]
        y[kn] -= sol[k]
        y[k] = (sol[kp] - sol[kn]) - vol.voltage(variables) * self.energy_factor

    def process_voltage_D(self, vol: Voltage, sol, D, variables):
        k = self.voltage_index(vol)
        kp = self.port_index(vol.p)
        kn = self.port_index(vol.n)
        D[kp][k]+=1
        D[kn][k]-=1
        D[k][kp] +=1
        D[k][kn] -=1

    def process_current_source_y(self, cs:  Current, sol, y, variables):
        cur = cs.get_amp(variables) * self.energy_factor
        y[self.port_index(cs.p)] += cur
        y[self.port_index(cs.n)] -= cur

    def process_current_source_D(self, cs: Current, sol, D, variables):
        # currents are constant
        pass

    def process_resistor_y(self, resi, sol, y, variables):
        G = 1/ resi.get_ohm(variables)
        pk = self.port_index(resi.p)
        nk = self.port_index(resi.n)
        current = (sol[pk] - sol[nk]) * G
        y[pk] -= current
        y[nk] += current

    def process_resistor_D(self, resi, sol, D, variables):
        pk = self.port_index(resi.p)
        nk = self.port_index(resi.n)
        G = 1/ resi.get_ohm(variables)
        D[pk][pk] -= G
        D[pk][nk] += G
        D[nk][pk] += G
        D[nk][nk] -= G

    def process_capacitor_y(self, c, sol, y, capa_voltages, variables):
        k = self.capa_index(c)
        pk = self.port_index(c.p)
        nk = self.port_index(c.n)
        if c.name in capa_voltages:
            v = capa_voltages[c.name] * self.energy_factor
            y[pk] += sol[k]
            y[nk] -= sol[k]
            y[k] = (sol[pk] - sol[nk]) - v
        else:
            y[k] = 0

    def process_capacitor_D(self, c, sol, D, capa_voltages, variables):
        k = self.capa_index(c)
        pk = self.port_index(c.p)
        nk = self.port_index(c.n)
        if c.name in capa_voltages:
            D[pk][k] += 1
            D[nk][k] += -1
            D[k][pk] +=1
            D[k][nk] -=1
        else:
            D[k][k] = 1

    def compute_y(self, sol, capa_voltages, variables):
        capa_voltages = capa_voltages or {}
        n = len(self.node_list) + len(self.voltage_list) + len(self.capa_list)
        y = np.zeros(n)
        for comp in self.netw.components.values():
            if isinstance(comp, Node):
                pass
            elif isinstance(comp, Voltage):
                self.process_voltage_y(comp, sol, y, variables)
            elif isinstance(comp, Current):
                self.process_current_source_y(comp, sol, y, variables)
            elif isinstance(comp, Resistor):
                self.process_resistor_y(comp, sol, y, variables)
            elif isinstance(comp, Diode):
                self.process_diode_y(comp, sol, y, variables)
            elif isinstance(comp, NPNTransistor):
                self.process_npn_transistor_y(comp, sol, y, variables)
            elif isinstance(comp, PNPTransistor):
                self.process_pnp_transistor_y(comp, sol, y, variables)
            elif isinstance(comp, Capacitor):
                self.process_capacitor_y(comp, sol, y, capa_voltages, variables)
            else:
                raise Exception("unknown component type of {0}".format(comp))

        # no euqation for ground, make sure its votage is 0
        k  = self.node_index(self.ground)
        y[k] = sol[k]
        return y

    def compute_D(self, sol, capa_voltages, variables):
        capa_voltages = capa_voltages or {}
        n = len(self.node_list) + len(self.voltage_list) + len(self.capa_list)
        D = np.zeros((n,n))
        for comp in self.netw.components.values():
            if isinstance(comp, Node):
                pass
            elif isinstance(comp, Voltage):
                self.process_voltage_D(comp, sol, D, variables)
            elif isinstance(comp, Current):
                self.process_current_source_D(comp, sol, D, variables)
            elif isinstance(comp, Resistor):
                self.process_resistor_D(comp, sol, D, variables)
            elif isinstance(comp, Diode):
                self.process_diode_D(comp, sol, D, variables)
            elif isinstance(comp, NPNTransistor):
                self.process_npn_transistor_D(comp, sol, D, variables)
            elif isinstance(comp, PNPTransistor):
                self.process_pnp_transistor_D(comp, sol, D, variables)
            elif isinstance(comp, Capacitor):
                self.process_capacitor_D(comp, sol, D, capa_voltages, variables)
            else:
                raise Exception("unknown component type of {0}".format(comp))

        # no euqation for ground, make sure its votage is 0
        k  = self.node_index(self.ground)
        D[k] = np.zeros(n)
        D[k][k] = 1
        return D

    def analyze(self,
                 maxit=20,
                 start_solution_vec=None,
                 abstol= 1e-8,
                 reltol= 1e-6,
                 variables=None,
                 capa_voltages=None,
                 start_voltages= None):
        if variables is None:
            variables = dict()
        n = len(self.node_list) + len(self.voltage_list) + len(self.capa_list)
        if start_solution_vec is None:
            if start_voltages is None:
                solution_vec0 = np.zeros(n)
            else:
                solution_vec0 = np.zeros(n)
                for vk in start_voltages:
                    p = self.netw.get_object(vk)
                    #                    p = self._port(vk)
                    n = self.port_index(p)
                    solution_vec0[n] = start_voltages[vk]
        else:
            solution_vec0 = start_solution_vec

        solution_vec = solution_vec0

        def f(x):
            return self.compute_y(x, capa_voltages, variables)

        def Df(x):
            return self.compute_D(x, capa_voltages, variables)

        res = solve(solution_vec, f, Df, abstol, reltol, maxit)
        if not isinstance(res, str):
            (sol, y, dfx, iterations, norm_y) = res
            norm_y = np.linalg.norm(y)
            return Result(self.netw, self, iterations, sol, variables, y, norm_y, np.linalg.cond(dfx,'fro'))

        alfa = 0.5

        for i in range(20):
            print(("energy factor ",i))
            alfa = (alfa + 1) / 2
            res = solve(solution_vec, f, Df, abstol, reltol, maxit, x0 = solution_vec0, alfa=alfa)
            if not isinstance(res, str):
                solution_vec = res[0]
                break
        if isinstance(res,str):
            print("failed getting initial solution")
            return res
        print("got initial solution, alfa={0}".format(alfa))

        while True:
            alfa = max(alfa / 1.1, 0)
            res = solve(solution_vec, f, Df, abstol, reltol, maxit, x0 = solution_vec0, alfa=alfa)
            if isinstance(res, str):
                print("alfa={0}".format(alfa))
                return res
            if alfa <=0:
                break
            else:
                solution_vec = res[0]

        (sol, y, dfx, iterations, norm_y) = res
        norm_y = np.linalg.norm(y)
        return Result(self.netw, self, iterations, sol, variables, y, norm_y, np.linalg.cond(dfx,'fro'))


    def transient(self,
                  maxtime,
                  timestep,
                  maxit=20,
                  start_solution_vec=None,
                  abstol= 1e-8,
                  reltol= 1e-6,
                  variables=None,
                  capa_voltages=None,
                  start_voltages=None):

        capa_voltages = capa_voltages or {}
        work_capa_voltages = dict()
        for comp in self.netw.components.values():
            if isinstance(comp, Capacitor):
                if not comp.name in capa_voltages:
                    work_capa_voltages[comp.name] = 0
                else:
                    work_capa_voltages[comp.name] = capa_voltages[comp.name]

        res = self.analyze(maxit=maxit,
                           start_solution_vec=start_solution_vec,
                           capa_voltages=work_capa_voltages,
                           variables=variables,
                           start_voltages=start_voltages)
        if isinstance(res, str):
            raise Exception("can not find inital solution")
        time = 0
        solutions= []
        sol = res.solution_vec
        solutions.append((time, res.get_voltages(), res.get_currents()))
        time += timestep

        while time < maxtime:
            for capa_name in work_capa_voltages:
                comp = self.netw.get_object(capa_name)
                capa = comp.capa
                current = res.get_current(comp.p)
                work_capa_voltages[capa_name] += timestep*current/capa
            res = self.analyze(maxit=maxit,
                              start_solution_vec=sol,
                              capa_voltages= work_capa_voltages,
                              variables = variables)
            if isinstance(res, str):
                break
            solutions.append((time, res.get_voltages(), res.get_currents()))
            sol = res.solution_vec
            time += timestep
        return solutions

""" simple routines for experimenting with nodal analysis """

import math
import pprint as pp
import numbers
import numpy as np

def explin(x: float, cutoff: float):
    if x < cutoff:
        return math.exp(x)
    return math.exp(cutoff) +  (x-cutoff) * math.exp(cutoff)

def dexplin(x:float, cutoff:float):
    if x < cutoff:
        return math.exp(x)
    return math.exp(cutoff)

def reldiff(x,y):
    return abs(x-y) /  max(abs(x), abs(y))

class Variable:
    """a variable"""
    def __init__(self, name, default=None):
        self.name = name
        self.default = default

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

    def get_currents(self, v, variables):
        raise NotImplementedError("ports method not implemented")


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

    def get_currents(self, v, variables):
        raise NotImplementedError("get_currents method not implemented")



class Node(Component):
    """a node, just a port in the network"""
    def __init__(self, parent, name):
        super().__init__(parent, name)
        self.port = Port(self, "port")

    def ports(self):
        return [self.port]

    def __repr__(self):
        return "<Node {0}>".format(self.name)

    def get_currents(self, v, variables):
        return {}


class Resistor(Node2):
    """resistor"""
    def __init__(self, parent, name, ohm):
        super().__init__(parent, name)
        self._ohm = ohm

    def get_ohm(self, variables):
        if isinstance(self._ohm, numbers.Number):
            return self._ohm
        if isinstance(self._ohm, Variable):
            v = variables.get(self._ohm.name, None)
            if v is None:
                raise Exception("did not find value for var {0}".format(self._ohm.name))
            return v
        raise Exception("bug")

    def __repr__(self):
        return "<Resistor {0}>".format(self.name)

    def get_currents(self, v,variables):
        vd = v[self.p] - v[self.n]
        i = vd / self.get_ohm(variables)
        return { self.p: i, self.n : -i}

class Current(Node2):
    """current source"""
    def __init__(self, parent, name, amp):
        super().__init__(parent, name)
        self.amp = amp

    def __repr__(self):
        return "<Current {0}>".format(self.name)

    def get_currents(self, v, variables):
        return {self.p: -self.amp, self.n: self.amp}



class Voltage(Node2):
    """current source"""
    def __init__(self, parent, name:str, volts:float):
        super().__init__(parent, name)
        assert isinstance(volts, (numbers.Number, Variable)), "volts must be a variable or a number"
        self.volts = volts

    def voltage(self, variables):
        if isinstance(self.volts, numbers.Number):
            return self.volts
        if isinstance(self.volts, Variable):
            v = variables.get(self.volts.name, None)
            if v is None:
                raise Exception("did not find value for var {0}".format(self.volts.name))
            return v
        raise Exception("bug")

    def __repr__(self):
        return "<Voltage {0}>".format(self.name)

    def get_currents(self, v, variables):
        return {}

class Diode(Node2):
    """solid state diode"""
    def __init__(self, parent, name, Is, Nut, max_current= 10):
        super().__init__(parent, name)
        self.Is = Is
        self.Nut = Nut
        self.max_current = max_current
        self.max_volt = self.volt_amp(max_current)

    def current(self, v):
        if v<0:
            return 0
        return self.Is * (math.exp(v/self.Nut)-1)

    def volt_amp(self, cur):
        if cur > 10:
            cur = 10
        if cur < 0:
            return 0
        return (math.log(cur/self.Is )+1)* self.Nut

    def diff_conductance(self, v):
        if v<0:
            return 0
        return self.Is * 1/self.Nut * math.exp(v/self.Nut)

    def __repr__(self):
        return "<Diode {0}>".format(self.name)

    def get_currents(self, v, variables):
        vd = v[self.p] - v[self.n]
        i = self.current(vd)
        return { self.p: i, self.n : -i}

class Capacitor(Node2):
    """ a capacitor"""

    def __init__(self, parent, name, capa):
        super().__init__(parent, name)
        self.capa = capa

    def get_currents(self, v, variables):
        return {}


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

        self.cutoff = cutoff

    def ports(self):
        return [self.B, self.C, self.E]

    def t1(self, vbe, vbc):
        return explin(vbe/self.VT, self.cutoff) - explin(vbc/self.VT, self.cutoff)

    def d_t1_vbe(self, vbe):
        return dexplin(vbe/self.VT, self.cutoff) / self.VT

    def d_t1_vbc(self, vbc):
        return -dexplin(vbc/self.VT, self.cutoff) / self.VT

    def t2(self, vbc):
        return 1/self.beta_R *(explin(vbc/self.VT, self.cutoff)-1)

    def d_t2_vbc(self, vbc):
        return 1/self.beta_R * dexplin(vbc/self.VT, self.cutoff) /self.VT

    def t3(self, vbe):
        return 1/self.beta_F *(explin(vbe/self.VT, self.cutoff)-1)

    def d_t3_vbe(self, vbe):
        return 1/self.beta_F * dexplin(vbe/self.VT, self.cutoff) / self.VT

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

    def get_currents(self, v, variables):
        vbc = v[self.B] - v[self.C]
        vbe = v[self.B] - v[self.E]
        return { self.B: self.IB(vbe, vbc),  self.E: -self.IE(vbe, vbc), self.C: self.IC(vbe, vbc)}



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
            d = Diode(self, name, comp.Is, comp.Nut, max_current = comp.max_current)
            self.components[name] = d
            return d
        if isinstance(comp, NPNTransistor):
            t = NPNTransistor(self, name, comp.IS, comp.VT, comp.beta_F, comp.beta_R)
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
    def __init__(self, network, analysis, iterations, solution_vec, variables):
        self.analysis = analysis
        self.network = network
        self.solution_vec = solution_vec
        self.voltages = dict()
        self.currents = dict()
        self.iterations = iterations
        for c in network.components.values():
            if isinstance(c, Capacitor):
                k = self.analysis.capa_index(c)
                cu = solution_vec[k]
                self.currents[c.p] = -cu
                self.currents[c.n] = cu
            for port in c.ports():
                k = self.analysis.port_index(port)
                self.voltages[port] = solution_vec[k]
            d = c.get_currents(self.voltages, variables)
            self.currents.update(d)

    def __repr__(self):
        return repr({"voltages": self.voltages, "currents": self.currents})

    def get_voltage(self, name_or_object):
        c = self._port(name_or_object)
        return self.voltages[c]

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


    def process_diode(self, comp, mat, r, solution_vec):
        if solution_vec is None:
            dv0 = comp.volt_amp(1)
        else:
            dv0 = self.voltage(solution_vec,comp.p) - self.voltage(solution_vec, comp.n)

        if dv0 > comp.max_volt:
            dv0 = comp.max_volt

        #pp.pprint(("d diode", dv0))

        Id = comp.current(dv0)
        dId = comp.diff_conductance(dv0)

        #I(dv) = I(v0) + dI(v0) * (dv- dv0)
        k = self.port_index(comp.p) #if port == comp.p:
        # current leaves node
        r[k] += Id
        r[k] -=  dId * dv0
        mat[k][self.port_index(comp.n)] += dId
        mat[k][k] -= dId
        k = self.port_index(comp.n)
        # current enters node
        r[k] -= Id
        r[k] +=  dId * dv0
        mat[k][self.port_index(comp.p)] += dId
        mat[k][k] -= dId

    def process_npn_transistor(self, tra, mat, r, solution_vec):
        #if port != tra.B:
        #    return
        #        print("-------------------------------------------------------")
        if solution_vec is None:
            vbe0 = 0 #   0.2
            vbc0 = 0
        else:
            vbe0 = self.voltage(solution_vec,tra.B) - self.voltage(solution_vec, tra.E)
            vbc0 = (self.voltage(solution_vec, tra.B) - self.voltage(solution_vec, tra.C))

        kB = self.port_index(tra.B)
        kE = self.port_index(tra.E)
        kC = self.port_index(tra.C)


        # IB(vbe, vbc) = IB(vbe0, vbc0) + d_IB_vbe(vbe0, vbc0) * (vbe -vbe0)
        #                               + d_IB_vbc(vbe0, vbc0) * (vbc- vbc0)
        # basis current leaves node

        r[kB] += tra.IB(vbe0, vbc0)
        r[kB] -= tra.d_IB_vbe(vbe0) * vbe0 + tra.d_IB_vbc(vbc0) * vbc0
        mat[kB][kB] += -tra.d_IB_vbe(vbe0)
        mat[kB][kB] += -tra.d_IB_vbc(vbc0)
        mat[kB][kE] += tra.d_IB_vbe(vbe0)
        mat[kB][kC] += tra.d_IB_vbc(vbc0)
        #pp.pprint((("VBE", vbe0), ("VBC", vbc0),
        #           ("B", tra.IB(vbe0, vbc0)),
        #           ("E", tra.IE(vbe0, vbc0)),
        #           ("C", tra.IC(vbe0, vbc0))))


        # IC(vbe, vbc) = IC(vbe0, vbc0) + d_IC_vbe(vbe0, vbc0) * (vbe -vbe0)
        #                               + d_IC_vbc(vbe0, vbc0) * (vbc- vbc0)
        # Collector current leaves nodes
        r[kC] += tra.IC(vbe0, vbc0)
        r[kC] -= tra.d_IC_vbe(vbe0) * vbe0 + tra.d_IC_vbc(vbc0) * vbc0
        mat[kC][kB] -= tra.d_IC_vbe(vbe0)
        mat[kC][kB] -= tra.d_IC_vbc(vbc0)
        mat[kC][kE] += tra.d_IC_vbe(vbe0)
        mat[kC][kC] += tra.d_IC_vbc(vbc0)
        #        pp.pprint(("CC", tra.d_IC_vbe(vbe0), tra.d_IC_vbc(vbc0)))

        # IE(vbe, vbc) = IE(vbe0, vbc0) + d_IE_vbe(vbe0, vbc0) * (vbe -vbe0)
        #                               + d_IE_vbc(vbe0, vbc0) * (vbc- vbc0)
        # emitter curren enters node
        r[kE] -= tra.IE(vbe0, vbc0)
        r[kE] += tra.d_IE_vbe(vbe0) * vbe0 + tra.d_IE_vbc(vbc0) * vbc0
        mat[kE][kB] += tra.d_IE_vbe(vbe0)
        mat[kE][kB] += tra.d_IE_vbc(vbc0)
        mat[kE][kE] -= tra.d_IE_vbe(vbe0)
        mat[kE][kC] -= tra.d_IE_vbc(vbc0)
        #pp.pprint(("EE", tra.d_IE_vbe(vbe0), tra.d_IE_vbc(vbc0)))

    def process_voltage(self, vol, mat, r, solution_vec, variables):
        k = self.voltage_index(vol)
        mat[k][self.port_index(vol.p)] = 1
        mat[k][self.port_index(vol.n)] = -1
        mat[self.port_index(vol.p)][k] = 1
        mat[self.port_index(vol.n)][k] = -1
        
        r[k] = vol.voltage(variables)

    def process_current_source(self, cs, mat, r, solution_vec, variables):
        r[self.port_index(cs.p)] -= cs.amp
        r[self.port_index(cs.n)] += cs.amp

    def process_resistor(self, resi, mat, r, soution_vec, variables):
          # I = (Vp - Vn) * G
        G = 1/ resi.get_ohm(variables)
        pk = self.port_index(resi.p)
        nk = self.port_index(resi.n)
        mat[pk][pk] -= G
        mat[pk][nk] += G
        mat[nk][pk] += G
        mat[nk][nk] -= G
        
    def compute_mat_and_r(self, solution_vec, capa_voltages, variables):
        capa_voltages = capa_voltages or {}
        n = len(self.node_list) + len(self.voltage_list) + len(self.capa_list)
        mat = np.zeros((n,n))
        r = np.zeros(n)

        for comp in self.netw.components.values():
            print(("comp", comp))
            if isinstance(comp, Node):
                pass
            elif isinstance(comp, Voltage):
                self.process_voltage(comp, mat, r, solution_vec, variables)
            elif isinstance(comp, Current):
                self.process_current_source(comp, mat, r, solution_vec, variables)
            elif isinstance(comp, Resistor):
                self.process_resistor(comp, mat, r, solution_vec, variables)
            elif isinstance(comp, Diode):
                self.process_diode(comp, mat, r, solution_vec)
            elif isinstance(comp, NPNTransistor):
                self.process_npn_transistor(comp, mat, r, solution_vec)
            else:
                raise Exception("unknown component type of {0}".format(comp))
            
        for c in self.capa_list:
            k = self.capa_index(c)
            mat[self.port_index(c.p)][k] = 1
            mat[self.port_index(c.n)] [k] = -1
            if c.name in capa_voltages:
                v = capa_voltages[c.name]
                mat[k][self.port_index(c.p)] = 1
                mat[k][self.port_index(c.n)] = -1
                r[k] = v
            else:
                mat[k][k] = 1
                r[k] = 0

        # for each node we create a row with a equation
        """for node in self.node_list:
            if node == self.ground:
                continue
            k = self.node_index(node)
            I = 0
            for port in self.node_ports(node):
                comp = port.component
                if isinstance(comp, Node):
                    pass
                elif isinstance(comp, Current):
                    pass
                elif isinstance(comp, Resistor):
                    pass
                    # I = (Vp - Vn) * G
                    #G = 1/ comp.get_ohm(variables)

                    #if port == comp.p:
                    #    op = comp.n
                    #else:
                    #    op = comp.p
                    #oi = self.port_index(op)
                    #mat[k][oi] += G
                    #mat[k][k] -= G
                elif isinstance(comp, Voltage):
                    pass
                    #kv = self.voltage_index(comp)
                    #if port == comp.p:
                        # current enters node
                    #    mat[k][kv] = 1
                    #else:
                     #   mat[k][kv] = -1
                elif isinstance(comp, Capacitor):
                    pass
                elif isinstance(comp, Diode):
                    pass #self.process_diode(k, comp, port, mat, r, solution_vec)
                elif isinstance(comp, NPNTransistor):
                    pass #self.process_npn_transistor(comp, port, mat, r, solution_vec)
                else:
                    raise Exception("unknown component type of {0}".format(comp))
            r[k] += I"""
        # we do not need the equation for ground, use this equation to fix voltage
        # ground voltage is fixed to 0
        k  = self.node_index(self.ground)
        mat[k] = np.zeros(n)
        mat[k][k] = 1
        r[k] = 0
        return (mat,r)

    def analyze(self,
                maxit=20,
                start_solution_vec=None,
                abstol= 1e-8,
                reltol= 1e-6,
                variables=None,
                capa_voltages=None,
                alpha = 1):
        if variables is None:
            variables = dict()

        solution_vec = start_solution_vec

        i = 0
        while True:
            if i >=maxit:
                for node in self.node_list:
                    i = self.node_index(node)
                    pp.pprint((i, node))#, self.solution_vec[i]))
                #print(mat)
                #print(r)
                print("no convergence {0} {1}".format(maxit, alpha))
                return "no_convergence"
            i += 1
            (mat,r) = self.compute_mat_and_r(solution_vec, capa_voltages, variables)
            print("--------- mat ----------")
            print(self.node_list)
            print(mat)
            print(r)
            solution_vec_n = np.linalg.solve(mat, r)
#            pp.pprint(("Solution", solution_vec_n))
            if solution_vec is not None:
                close_enough = True
                for j in range(solution_vec.size):
                    x = solution_vec[j]
                    y = solution_vec_n[j]
                    if not (abs(x-y) < abstol  or reldiff(x,y) < reltol ):
                        close_enough = False
                if close_enough:
                    iterations = i
                    break
            if solution_vec is None:
                solution_vec = solution_vec_n
            else:
                solution_vec = solution_vec_n * alpha + (1- alpha) * solution_vec

        self.mat = mat
        self.r = r
        self.solution_vec = solution_vec
        return Result(self.netw, self, iterations, self.solution_vec, variables)

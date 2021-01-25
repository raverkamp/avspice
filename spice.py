""" simple routines for experimenting with nodal analysis """

import math
import pprint as pp
import numbers
import numpy as np
import sympy as sp

def explin(x, cutoff):
    if x < cutoff:
        return math.exp(x)
    return math.exp(cutoff) +  (x-cutoff) * math.exp(cutoff)

def dexplin(x, cutoff):
    if x < cutoff:
        return math.exp(x)
    return math.exp(cutoff)

def reldiff(x,y):
    return abs(x-y) /  min(abs(x), abs(y))

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

    def get_currents(self, v):
        raise NotImplementedError("ports method not implemented")


class Port:
    """ components are connected via their ports"""

    def __init__(self, component, name):
        self.component = component
        self.name = name

    def __repr__(self):
        return "<Port {0}.{1}>".format(self.component.name, self.name)

    def pname(self):
        """printable name of the port"""
        return self.component.name + "." + self.name


class Node2(Component):
    """a component with just two ports"""

    def __init__(self, parent, name):
        super().__init__(parent, name)
        self.p = Port(self, "p")
        self.n = Port(self, "n")

    def ports(self):
        return [self.p, self.n]

    def get_currents(self, v):
        raise NotImplementedError("ports method not implemented")



class Node(Component):
    """a node, just a port in the network"""
    def __init__(self, parent, name):
        super().__init__(parent, name)
        self.port = Port(self, "port")

    def ports(self):
        return [self.port]

    def __repr__(self):
        return "<Node {0}>".format(self.name)

    def get_currents(self, v):
        return {}


class Resistor(Node2):
    """resistor"""
    def __init__(self, parent, name, ohm):
        super().__init__(parent, name)
        self.ohm = ohm

    def __repr__(self):
        return "<Resistor {0}>".format(self.name)

    def get_currents(self, v):
        vd = v[self.p] - v[self.n]
        i = vd / self.ohm
        return { self.p: i, self.n : -i}

class Current(Node2):
    """current source"""
    def __init__(self, parent, name, amp):
        super().__init__(parent, name)
        self.amp = amp

    def __repr__(self):
        return "<Current {0}>".format(self.name)

    def get_currents(self, v):
        return {self.p: -self.amp, self.n: self.amp}



class Voltage(Node2):
    """current source"""
    def __init__(self, parent, name, volts):
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

    def get_currents(self, v):
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

    def get_currents(self, v):
        vd = v[self.p] - v[self.n]
        i = self.current(vd)
        return { self.p: i, self.n : -i}

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

    def __init__(self, parent, name, IS, VT, beta_F, beta_R, cutoff=40):
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

    def get_currents(self, v):
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

    def addR(self, name, ohm):
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
    adj = dict()
    adj[nw.ground.port] = []
    def add(p1, p2):
        if p1 in adj:
            l = adj[p1]
            l.append(p2)
        else:
            adj[p1] = [p2]

    for (p1, p2) in nw.connections:
        add(p1, p2)
        add(p2, p1)

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
    def __init__(self, network, analysis, iterations, solution_vec):
        self.analysis = analysis
        self.network = network
        self.solution_vec = solution_vec
        self.voltages = dict()
        self.currents = dict()
        self.iterations = iterations
        for c in network.components.values():
            for port in c.ports():
                k = self.analysis.port_index(port)
                self.voltages[port] = solution_vec[k]
            d = c.get_currents(self.voltages)
            self.currents.update(d)

    def get_voltage(self, name_or_object):
        c = self._port(name_or_object)
        return self.voltages[c]

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

    def voltage(self, solution_vec, port):
        k = self.port_index(port)
        return solution_vec[k]


    def process_diode(self, k, comp, port, mat, r, solution_vec):
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
        if port == comp.p:
            # current leaves node
            r[k] += Id
            r[k] -=  dId * dv0
            mat[k][self.port_index(comp.n)] += dId
            mat[k][k] -= dId
        elif port == comp.n:
            # current enters node
            r[k] -= Id
            r[k] +=  dId * dv0
            mat[k][self.port_index(comp.p)] += dId
            mat[k][k] -= dId

    def process_npn_transistor(self, tra, port, mat, r, solution_vec):
        if port != tra.B:
            return
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

    def compute_mat_and_r(self, solution_vec, variables):
        n = len(self.node_list) + len(self.voltage_list)
        mat = np.zeros((n,n))
        r = np.zeros(n)
        # ground voltage is fixed to 0
        k  = self.node_index(self.ground)
        mat[k][k] = 1
        r[k] = 0

         # equations for voltage sources
        for vol in self.voltage_list:
            k = self.voltage_index(vol)
            mat[k][self.port_index(vol.p)] = 1
            mat[k][self.port_index(vol.n)] = -1
            r[k] = vol.voltage(variables)

        for node in self.node_list:
            if node == self.ground:
                continue
            k = self.node_index(node)
            I = 0
            GG = 0
            for port in self.node_ports(node):
                comp = port.component
                if isinstance(comp, Node):
                    pass
                elif isinstance(comp, Current):

                    if comp.p == port:
                        # current enters node
                        I = I - comp.amp
                    else:
                        # current leaves node
                        I = I + comp.amp
                elif isinstance(comp, Resistor):
                    # I = (Vp - Vn) * G
                    G = 1/ comp.ohm

                    if port == comp.p:
                        op = comp.n
                    else:
                        op = comp.p
                    oi = self.port_index(op)
                    mat[k][oi] += G
                    mat[k][k] -= G
                elif isinstance(comp, Voltage):
                    kv = self.voltage_index(comp)
                    if port == comp.p:
                        # current enters node
                        mat[k][kv] = 1
                    else:
                        mat[k][kv] = -1
                elif isinstance(comp, Diode):
                    self.process_diode(k, comp, port, mat, r, solution_vec)
                elif isinstance(comp, NPNTransistor):
                    self.process_npn_transistor(comp, port, mat, r, solution_vec)
                else:
                    raise Exception("unknown component type of {0}".format(comp))
            mat[k][k] += GG
            r[k] += I
        return (mat,r)

    def analyze(self,
                maxit=20,
                start_solution_vec=None,
                abstol= 1e-8,
                reltol= 1e-6,
                variables=None):
        if variables is None:
            variables = dict()

        solution_vec = start_solution_vec

        for i in range(10000):
            if i >=maxit:
                for node in self.node_list:
                    i = self.node_index(node)
                    pp.pprint((i, node))#, self.solution_vec[i]))
                print(mat)
                print(r)
                return "no_convergence"

            (mat,r) = self.compute_mat_and_r(solution_vec, variables)
            #print(self.node_list)
            #            print("--------- mat ----------")
            #print(mat)
             #print(r)

            solution_vec_n = np.linalg.solve(mat, r)
#            pp.pprint(("Solution", solution_vec_n))
            if solution_vec is not None:
                close_enough = True
                for j in range(len(solution_vec)):
                    x = solution_vec[j]
                    y = solution_vec_n[j]
                    if abs(x-y) > abstol and reldiff(x,y) > reltol:
                        close_enough = False
                if close_enough:
                    iterations = i
                    break
            solution_vec = solution_vec_n

        self.mat = mat
        self.r = r
        self.solution_vec = solution_vec
        return Result(self.netw, self, iterations, self.solution_vec)


def symadd(l):
    """convert a list of sympy expressions into a sympy sum"""
    if not l:
        return sp.sympify(0)
    res = None
    for x in l:
        if res :
            res = sp.Add(res, x)
        else:
            res = x
    return res

def sym_analyze(net):
    nodes = compute_nodes(net)
    nnodes = len(nodes)


    voltages = []
    for comp in net.components.values():
        if isinstance(comp, Voltage):
            voltages.append(comp)

    def voltage_index(voltage):
        return voltages.index(voltage) + nnodes


    def p_2_n_index(port):
        for i in range(nnodes):
            if port in nodes[i].ports:
                return i
        raise Exception("BUG")

    # kirchhoff for all nodes
    volts = []
    for node in nodes:
        sy = sp.symbols(node.name)
        volts.append(sy)

    for voltage in voltages:
        sy = sp.symbols(voltage.name)
        volts.append(sy)


    equations = []
    # ground is zero
    equations.append(volts[0])

    for vol in voltages:
        k = voltage_index(vol)
        p1 = vol.p
        p2 = vol.n
        n1 = p_2_n_index(p1)
        n2 = p_2_n_index(p2)
        equations.append(volts[n1] - volts[n2] - vol.volts)

    for i in range(1,nnodes):
        node = nodes[i]
        summand_list = []
        for port in node.ports:
            comp = port.component
            if isinstance(comp, Node):
                pass
            elif isinstance(comp, Current):
                if comp.p == port:
                    I = - comp.amp
                else:
                    I = comp.amp
                summand_list.append(sp.sympify(I))
            elif isinstance(comp, Resistor):
                o = comp.ohm
                if port == comp.p:
                    op = comp.n
                else:
                    op = comp.p

                oi = p_2_n_index(op)
                summand_list.append((volts[i] - volts[oi]) / sp.sympify(o))
            elif isinstance(comp, Voltage):
                if port == comp.p:
                    ss = -1
                else:
                    ss = 1
                k = voltage_index(comp)

                summand_list.append(sp.Mul(volts[k],ss))
            else:
                raise Exception("unknown component type of {0}".format(comp))
        equations.append(symadd(summand_list))
    print(equations)
    a = sp.solvers.nsolve(equations, volts, [0]* nnodes)
    print(a)

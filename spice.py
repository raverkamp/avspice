""" simple routines for experimenting with nodal analysis """

import math
import pprint as pp
import numpy as np
import sympy as sp

class Component:
    """Componet in a electrical network, e.g. resistor, current source, node"""
    def __init__(self, parent, name):
        self.name = name
        self.parent = parent

    def ports(self):
        """return the ports of this component"""
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
        self.ohm = ohm

    def __repr__(self):
        return "<Resistor {0}>".format(self.name)


class Current(Node2):
    """current source"""
    def __init__(self, parent, name, amp):
        super().__init__(parent, name)
        self.amp = amp

    def __repr__(self):
        return "<Current {0}>".format(self.name)


class Voltage(Node2):
    """current source"""
    def __init__(self, parent, name, volts):
        super().__init__(parent, name)
        self.volts = volts

    def __repr__(self):
        return "<Voltage {0}>".format(self.name)

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

    def compute_mat_and_r(self, solution_vec):
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
            r[k] = vol.volts

        for node in self.node_list:
            if node == self.ground:
                continue
            k = self.node_index(node)
            I = 0
            GG = 0
            for port in self.node_ports(node):
                comp = port.component
                pp.pprint(("comp", comp))
                if isinstance(comp, Node):
                    pass
                elif isinstance(comp, Current):
                    if comp.p == port:
                        I = I + comp.amp
                    else:
                        I = I - comp.amp
                elif isinstance(comp, Resistor):
                    o = comp.ohm
                    GG = GG + 1/o
                    if port == comp.p:
                        op = comp.n
                    else:
                        op = comp.p
                    oi = self.port_index(op)
                    mat[k][oi] = - 1/o
                elif isinstance(comp, Voltage):
                    if port == comp.p:
                        ss = -1
                    else:
                        ss = 1
                    kv = self.voltage_index(comp)
                    mat[k][kv] = ss
                elif isinstance(comp, Diode):
                    if port == comp.p:
                        ss = 1
                        op = comp.n
                    else:
                        ss = -1
                        op = comp.p
                    if solution_vec is None:
                        dv = comp.volt_amp(1)
                    else:
                        dv = (solution_vec[self.port_index(comp.p)]
                              - solution_vec[self.port_index(comp.n)])
                    if dv > comp.max_volt:
                        dv = comp.max_volt
                    curr = comp.current(dv)
                    conductance = comp.diff_conductance(dv)

                    pp.pprint((comp, dv, curr, conductance))

                    I = I - curr * ss # base current
                    I = I + conductance * dv * ss
                    GG = GG + conductance
                    oi = self.port_index(op)
                    mat[k][oi] = - conductance
                else:
                    raise Exception("unknown component type of {0}".format(comp))
            mat[k][k] = GG
            r[k] = I
        return (mat,r)

    def analyze(self):
        self.node_list = compute_nodes(self.netw)
        self.ground = self.node_list[0]

        for node in self.node_list:
            for port in node.ports:
                self._port_to_node[port] = node

        for comp in self.netw.components.values():
            if isinstance(comp, Voltage):
                self.voltage_list.append(comp)

        solution_vec = None

        for i in range(100):
            (mat,r) = self.compute_mat_and_r(solution_vec)
            solution_vec_n = np.linalg.solve(mat, r)
            if solution_vec is not None:
                diff = np.linalg.norm(solution_vec-solution_vec_n)
                if diff < 1e-6:
                    solution_vec = solution_vec_n
                    break
                pp.pprint(("diff", i, diff))
                pp.pprint(solution_vec)
                pp.pprint(solution_vec_n)

            solution_vec = solution_vec_n

        self.mat = mat
        self.r = r
        self.solution_vec = solution_vec
        return self.extract_result()


    def extract_result(self):
        voltages = dict()
        resistors = {}
        diodes = {}
        for comp in self.netw.components.values():
            if isinstance(comp, Node):
                voltages[comp] = self.solution_vec[self.port_index(comp.port)]
            if isinstance(comp, Resistor):
                ipp = self.port_index(comp.p)
                inp = self.port_index(comp.n)
                dv = self.solution_vec[ipp] - self.solution_vec[inp]
                curr = dv / comp.ohm
                resistors[comp] = (dv,curr)
            if isinstance(comp, Diode):
                ipp = self.port_index(comp.p)
                inp = self.port_index(comp.n)
                dv = self.solution_vec[ipp] - self.solution_vec[inp]
                curr = comp.current(dv)
                diodes[comp] = (dv, curr)

        return (voltages, resistors, diodes)






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

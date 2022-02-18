""" simple routines for experimenting with nodal analysis """

import math
import pprint as pp
import numbers
import numpy as np
import solving
import util

class Variable:
    """a variable"""
    def __init__(self, name, default=None):
        self.name = name
        assert default is None or isinstance(default, numbers.Number)
        self.default = default

    def __repr__(self):
        return f"<Variable {self.name}, {self.default}>"

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
        raise Exception(f"Component {self.name} does not have port {name}")

    def get_val(self, var_or_num, variables):
        if isinstance(var_or_num, numbers.Number):
            return var_or_num
        if isinstance(var_or_num, Variable):
            val = variables.get(var_or_num.name, None)
            if val is None:
                if var_or_num.default is None:
                    raise Exception(f"did not find value for var {var_or_num} in {self.name}")
                return var_or_num.default
            return val
        raise Exception("bug")

class Port:
    """ components are connected via their ports"""

    def __init__(self, component:Component, name:str):
        self.component = component
        self.name = name

    def __repr__(self):
        return f"<Port {self.component.name}.{self.name}>"

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

class Node(Component):
    """a node, just a port in the network"""
    def __init__(self, parent, name):
        super().__init__(parent, name)
        self.port = Port(self, "port")

    def ports(self):
        return [self.port]

    def __repr__(self):
        return f"<Node {self.name}>"


class Resistor(Node2):
    """resistor"""
    def __init__(self, parent, name, ohm):
        super().__init__(parent, name)
        self._ohm = ohm

    def get_ohm(self, variables):
        return self.get_val(self._ohm, variables)

    def __repr__(self):
        return f"<Resistor {self.name}>"

class Current(Node2):
    """current source"""
    def __init__(self, parent, name, amp):
        super().__init__(parent, name)
        self.amp = amp

    def __repr__(self):
        return f"<Current {self.name}>"

    def get_amp(self, variables):
        return self.get_val(self.amp, variables)

    def code(self,cname):
        return ([], ([],f"{self.amp}"))

class Voltage(Node2):
    """voltage source"""
    def __init__(self, parent, name:str, volts:float):
        super().__init__(parent, name)
        assert isinstance(volts, (numbers.Number, Variable)), "volts must be a variable or a number"
        self._volts = volts

    def voltage(self, time, variables):
        v = self.get_val(self._volts, variables)
        return v

    def __repr__(self):
        return f"<Voltage {self.name}>"

    def code(self, name, variables):
        v = self.get_val(self._volts, variables)
        init = [f"self.{name} = NVoltage({v})"]
        voltage = ([f"{name}_voltage = self.{name}.voltage(time)"],f"{name}_voltage")
        return (init, voltage)

class SineVoltage(Voltage):

    def __init__(self, parent, name:str, volts:float, frequency: float):
        super().__init__(parent, name, volts)
        self._frequency = frequency

    def voltage(self, time, variables):
        v = self.get_val(self._volts, variables)
        return v * math.sin(2 * math.pi * self._frequency)

    def code(self, name, variables):
        v = self.get_val(self._volts, variables)
        init = [f"self.{name} = NSineVoltage({v}, {self._frequency})"]
        voltage = ([f"{name}_voltage = self.{name}.voltage(time)"],f"{name}_voltage")
        return (init, voltage)

class SawVoltage(Voltage):

    def __init__(self, parent, name:str, volts:float, frequency: float):
        super().__init__(parent, name, volts)
        self._frequency = frequency

    def voltage(self, time, variables):
        v = self.get_val(self._volts, variables)
        return v * util.saw_tooth(1,time)

    def code(self, name, variables):
        v = self.get_val(self._volts, variables)
        init = [f"self.{name} = NSawVoltage({v}, {self._frequency})"]
        voltage = ([f"{name}_voltage = self.{name}.voltage(time)"],f"{name}_voltage")
        return (init, voltage)


class Diode(Node2):
    """solid state diode"""
    def __init__(self, parent, name, Is, Nut, lcut_off = -40, rcut_off=40):
        super().__init__(parent, name)
        self.Is = Is
        self.Nut = Nut

        self.lcut_off = lcut_off
        self.rcut_off = rcut_off

    def code(self, cname, dvname):
        init = [f"self.{cname} = NDiode({self.Is},{self.Nut},{self.lcut_off},{self.rcut_off})"]
        curr = ([], f"self.{cname}.current({dvname})")
        dcurr = ([], f"self.{cname}.diff_current({dvname})")
        return (init, curr, dcurr)

    def __repr__(self):
        return f"<Diode {self.name}>"

class Capacitor(Node2):
    """ a capacitor"""

    def __init__(self, parent, name, capa):
        super().__init__(parent, name)
        self._capa = capa

    def get_capa(self, variables):
        a = self.get_val(self._capa, variables)
        return a

    def __repr__(self):
        return f"<Capacitor {self.name}>"


class Inductor(Node2):
    """ a Spule" """

    def __init__(self, parent, name, induc):
        super().__init__(parent, name)
        self.induc = induc

    def __repr__(self):
        return f"<Inductor {self.name}>"



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

    def code(self, name, vb, ve, vc):
        prefix = name
        me = "self." + prefix + "_"
        initt = [f"{me} = NNPNTransistor({self.IS}, {self.VT},"
                 +f" {self.beta_F}, {self.beta_R}, {self.lcutoff}, {self.rcutoff})"]
        vbe = f"{prefix}_vbe"
        vbc = f"{prefix}_vbc"

        cinit =[f"{vbe} = {vb}- {ve}",
                f"{vbc} = {vb}- {vc}"]
        curr = (f"-{me}.IB({vbe}, {vbc})",
                f"{me}.IE({vbe}, {vbc})",
                f"-{me}.IC({vbe}, {vbc})")

        dinit = cinit

        d = ((f"-{me}.d_IB_vbe({vbe})- {me}.d_IB_vbc({vbc})",
             f"{me}.d_IB_vbe({vbe})",
             f"{me}.d_IB_vbc({vbc})"),

             (f"{me}.d_IE_vbe({vbe}) + {me}.d_IE_vbc({vbc})",
              f"-{me}.d_IE_vbe({vbe})",
              f"-{me}.d_IE_vbc({vbc})"),

             (f"-{me}.d_IC_vbe({vbe}) - {me}.d_IC_vbc({vbc})",
              f"{me}.d_IC_vbe({vbe})",
              f"{me}.d_IC_vbc({vbc})"))

        return (initt, (cinit, curr), (dinit,d))


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
                 cutoff:float=40):
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

    def code(self, name, vb, ve, vc):
        prefix = name
        me = "self." + prefix + "_"
        initt = [f"{me} = NPNPTransistor({self.IS}, {self.VT}, {self.beta_F},"
                 + f" {self.beta_R}, {self.lcutoff}, {self.rcutoff})"]
        vbe = f"{prefix}_vbe"
        vbc = f"{prefix}_vbc"

        cinit =[f"{vbe} = {vb}- {ve}",
                f"{vbc} = {vb}- {vc}"]
        curr = (f"-{me}.IB({vbe}, {vbc})",
                f"-{me}.IE({vbe}, {vbc})",
                f"-{me}.IC({vbe}, {vbc})")

        dinit = cinit

        d = ((f"-{me}.d_IB_vbe({vbe})- {me}.d_IB_vbc({vbc})",
             f"{me}.d_IB_vbe({vbe})",
             f"{me}.d_IB_vbc({vbc})"),

             (f"-{me}.d_IE_vbe({vbe}) - {me}.d_IE_vbc({vbc})",
              f"{me}.d_IE_vbe({vbe})",
              f"{me}.d_IE_vbc({vbc})"),

             (f"-{me}.d_IC_vbe({vbe}) - {me}.d_IC_vbc({vbc})",
              f"{me}.d_IC_vbe({vbe})",
              f"{me}.d_IC_vbc({vbc})"))

        return (initt, (cinit, curr), (dinit,d))


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
            raise Exception(f"Name {name} already exists")
        r = Resistor(self, name , ohm)
        self.components[name] = r
        return r

    def addC(self, name, amp):
        """add a curent source"""
        if name in self.components:
            raise Exception(f"Name {name} already exists")
        c = Current(self, name, amp)
        self.components[name] = c
        return c

    def addV(self, name, volts):
        """add a voltage source"""
        if name in self.components:
            raise Exception(f"Name {name} already exists")
        v = Voltage(self, name, volts)
        self.components[name] = v
        return v

    def addSineV(self, name, volts, frequency):
        """add a voltage source"""
        if name in self.components:
            raise Exception(f"Name {name} already exists")
        v = SineVoltage(self, name, volts, frequency)
        self.components[name] = v
        return v

    def addSawV(self, name, volts, frequency):
        """add a voltage source"""
        if name in self.components:
            raise Exception(f"Name {name} already exists")
        v = SawVoltage(self, name, volts, frequency)
        self.components[name] = v
        return v


    def addN(self,name):
        """add a node"""
        if name in self.components:
            raise Exception(f"Name {name} already exists")
        node = Node(self, name)
        self.components[name] = node
        return node

    def addD(self, name, Is, Nut):
        if name in self.components:
            raise Exception(f"Name {name} already exists")
        d = Diode(self, name, Is, Nut)
        self.components[name] = d
        return d

    def addComp(self, name, comp):
        if name in self.components:
            raise Exception(f"Name {name} already exists")
        if isinstance(comp, Diode):
            d = Diode(self,
                      name,
                      comp.Is,
                      comp.Nut,
                      lcut_off = comp.lcut_off,
                      rcut_off = comp.rcut_off)
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
        raise Exception(f"addComp not supported for {comp}")

    def addCapa(self, name, capa):
        if name in self.components:
            raise Exception("fName {name} already exists")
        c = Capacitor(self, name, capa)
        self.components[name] = c
        return c

    def addInduc(self, name, induc):
        if name in self.components:
            raise Exception("Name {name} already exists")
        indu = Inductor(self, name, induc)
        self.components[name] = indu
        return indu

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
            raise Exception(f"unknown component: {name}")
        c = self.components[name]
        if len(l) == 1:
            return c
        if len(l) == 2:
            return c.get_port(l[1])
        raise Exception(f"too many components in name {name}")

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
        return f"<XNode {self.name}: {self.ports}>"

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
    adj = {}
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
                raise Exception(f"{x} is not connected")
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
        raise Exception(f"some ports are not connected to ground: {allports}")
    return nodes

class Result:
    """result of an analysis run"""
    def __init__(self, network, analysis, iterations, solution_vec, y, y_norm, mat_cond, currents):
        self.analysis = analysis
        self.network = network
        self.solution_vec = solution_vec
        self.voltages = {}
        self.currents = {}
        self.iterations = iterations
        self.mat_cond = mat_cond
        self.y = y
        self.y_norm = y_norm
        self.currents = currents

        for c in network.components.values():
            for port in c.ports():
                k = self.analysis.port_index(port)
                self.voltages[port] = solution_vec[k]

    def get_voltages(self):
        res = {}
        for (p,v) in self.voltages.items():
            res[p.pname()] = v
        return res

    def get_currents(self):
        return self.currents

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
            raise Exception(f"{name_or_object} is not a Node2 or name for one")
        return self.voltages[c.p] - self.voltages[c.n]

    def _port(self, name_or_comp):
        c = None
        if isinstance(name_or_comp, str):
            c = self.network.get_object(name_or_comp)
        else:
            c = name_or_comp
        if not isinstance(c, Port):
            raise Exception(f"not a port or node or name thereof: {name_or_comp}")
        return c

    def get_current(self, port):
        if isinstance(port, Port):
            port  = port.pname()
        return self.currents[port]

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
                    print(port.pname() + " " + str(self.get_current(port)))

class CodeGenerator:
    def __init__(self, n, n_curr_ports):
        self.n = n
        self.n_curr_ports = n_curr_ports
        self.init = [
            "def bla():",
            "   return Computer()",
            "",
            "class Computer:",
            "    def __init__(self):",
            "        from ncomponents import NDiode, NNPNTransistor, NPNPTransistor,"
                    + "NVoltage, NSineVoltage, NSawVoltage"]
        self.y_code = ["    def y(self, time, sol, state_vec):",
                       "        import numpy as np",
                       f"        res = np.zeros({self.n})"]

        self.dy_code = ["    def dy(self, time, sol, state_vec):",
                        "        import numpy as np",
                        f"        res = np.zeros(({self.n},{self.n}))"]
        self.cur_code  = ["    def currents(self, time, sol, state_vec):",
                          "        import numpy as np",
                          f"        res = np.zeros({self.n_curr_ports})"]
        self.ysum=[]
        for i in range(self.n):
            self.ysum.append([])
        self.dysum=[]
        for i in range(self.n):
            x = []
            for j in range(n):
                x.append([])
            self.dysum.append(x)

    def add_ysum(self, k, e):
        self.ysum[k].append(e)

    def add_dysum(self, k, j, e):
        self.dysum[k][j].append(e)

    def add_to_method(self, m, lines_or_str):
        if isinstance(lines_or_str, str):
            l=[lines_or_str]
        else:
            l=lines_or_str
        for x in l:
            assert isinstance(x, str)
            m.append("        " + x)

    def add_to_init(self,l):
        self.add_to_method(self.init, l)

    def add_to_y_code(self, l):
        self.add_to_method(self.y_code, l)

    def add_to_dy_code(self, l):
        self.add_to_method(self.dy_code, l)

    def add_to_cur_code(self, l):
        self.add_to_method(self.cur_code, l)

class Analysis:
    """captures all data for analysis"""

    def __init__(self, netw):
        self.netw = netw
        self._port_to_node = {}
        self.node_list = []
        self.voltage_list = []
        self.capa_list = []
        self.induc_list = []
        self.ground = None
        self.mat = None
        self.r = None
        self.solution_vec = None
        self.node_list = compute_nodes(self.netw)
        self.ground = self.node_list[0]
        self.curr_port_list = []


        for node in self.node_list:
            for port in node.ports:
                self._port_to_node[port] = node

        for comp in self.netw.components.values():
            if isinstance(comp, Voltage):
                self.voltage_list.append(comp)
            if isinstance(comp, Capacitor):
                self.capa_list.append(comp)
            if isinstance(comp, Inductor):
                self.induc_list.append(comp)
            self.curr_port_list.extend(comp.ports())

    def node_index(self, node):
        return self.node_list.index(node)

    def port_index(self, port):
        node = self._port_to_node[port]
        return self.node_index(node)

    def voltage_index(self, voltage):
        return self.voltage_list.index(voltage) + len(self.node_list)

    def capa_index(self, capa):
        return self.capa_list.index(capa) + len(self.node_list) + len(self.voltage_list)

    def induc_index(self, induc):
        return (self.induc_list.index(induc)
                + len(self.node_list)
                + len(self.voltage_list)
                + len(self.capa_list))

    def state_index(self, x):
        if isinstance(x, Capacitor):
            return self.capa_list.index(x)
        if isinstance(x, Inductor):
            return self.induc_list.index(x) + len(self.capa_list)
        raise Exception(f"no state index for component {x}")

    def state_size(self):
        return len(self.capa_list) + len(self.induc_list)


    def _equation_size(self):
        return (len(self.node_list)
                + len(self.voltage_list)
                + len(self.capa_list)
                + len(self.induc_list))

    def curr_index(self, p):
        return self.curr_port_list.index(p)

    def generate_code(self, variables, transient):
        cg = CodeGenerator(self._equation_size(), len(self.curr_port_list))
        n = self._equation_size()

        counter = 0
        for comp in self.netw.components.values():
            cname = comp.name + str(counter)

            if isinstance(comp, Node2):
                kp = self.port_index(comp.p)
                kn = self.port_index(comp.n)
            if isinstance(comp, Node):
                pass
            elif isinstance(comp, Voltage):
                (vinit, (pre , expr)) = comp.code(cname, variables)
                cg.add_to_init(vinit)
                k = self.voltage_index(comp)
                cg.add_to_y_code(pre)
                cg.add_ysum(kp, f"sol[{k}]")
                cg.add_ysum(kn, f"(-sol[{k}])")
                cg.add_ysum(k,f"(sol[{kp}] - sol[{kn}]) - ({expr})")
                cg.add_dysum(kp, k, "1")
                cg.add_dysum(kn, k, "-1")
                cg.add_dysum(k, kp, "1")
                cg.add_dysum(k, kn, "-1")

                cg.add_to_cur_code([f"res[{self.curr_index(comp.p)}] = sol[{k}]",
                                    f"res[{self.curr_index(comp.n)}] = -(sol[{k}])"])

            elif isinstance(comp, Current):
                (cinit, (pre, expr)) = comp.code(cname)
                cg.add_to_init(cinit)
                cg.add_to_y_code(pre)
                cg.add_ysum(kp, f"({expr})")
                cg.add_ysum(kn, f"(-({expr}))")

                cg.add_to_cur_code(pre)
                cg.add_to_cur_code([f"res[{self.curr_index(comp.p)}] = -({expr})",
                                 f"res[{self.curr_index(comp.n)}] = {expr}"])

            elif isinstance(comp, Resistor):
                G = 1/ comp.get_ohm(variables)
                name = f"current_{comp.name}"
                cg.add_to_y_code([f"{name} = (sol[{kp}] - sol[{kn}]) * {G}"])
                cg.add_ysum(kp, f"(-{name})")
                cg.add_ysum(kn, f"{name}")

                cg.add_dysum(kp, kp, f"(-{G})")
                cg.add_dysum(kp, kn, f"{G}")
                cg.add_dysum(kn, kp, f"{G}")
                cg.add_dysum(kn, kn, f"(-{G})")

                cg.add_to_cur_code([f"{name} = (sol[{kp}] - sol[{kn}]) * {G}",
                                    f"res[{self.curr_index(comp.p)}] = ({name})",
                                    f"res[{self.curr_index(comp.n)}] =  -({name})"])

            elif isinstance(comp, Diode):
                (init_d, (cinit, curr), (dinit,dcurr)) = comp.code(cname,f"sol[{kp}]- sol[{kn}]")
                cg.add_to_init(init_d)
                cg.add_to_y_code(cinit)
                cg.add_to_dy_code(dinit)
                cg.add_ysum(kp, f"(-{curr})")
                cg.add_ysum(kn, f"({curr})")

                cg.add_dysum(kp, kp, f"(-{dcurr})")
                cg.add_dysum(kp, kn, f"{dcurr}")
                cg.add_dysum(kn, kp, f"{dcurr}")
                cg.add_dysum(kn, kn, f"(-{dcurr})")

                cg.add_to_cur_code(cinit)
                cg.add_to_cur_code([f"res[{self.curr_index(comp.p)}] = {curr}",
                                  f"res[{self.curr_index(comp.n)}] =  -({curr})"])

            elif isinstance(comp, (NPNTransistor, PNPTransistor)):
                kb = self.port_index(comp.B)
                ke = self.port_index(comp.E)
                kc = self.port_index(comp.C)

                (init_t, (cinit, (cb,ce,cc)),
                         (dinit,((dbb, dbe, dbc),
                                 (deb, dee, dec),
                                 (dcb, dce, dcc)))) = \
                     comp.code(cname,f"sol[{kb}]", f"sol[{ke}]", f"sol[{kc}]")

                cg.add_to_init(init_t)
                cg.add_to_y_code(cinit)
                cg.add_ysum(kb, f"({cb})")
                cg.add_ysum(ke, f"({ce})")
                cg.add_ysum(kc, f"({cc})")

                cg.add_to_dy_code(dinit)

                cg.add_dysum(kb, kb, f"({dbb})")
                cg.add_dysum(kb, ke, f"({dbe})")
                cg.add_dysum(kb, kc, f"({dbc})")

                cg.add_dysum(ke, kb, f"({deb})")
                cg.add_dysum(ke, ke, f"({dee})")
                cg.add_dysum(ke, kc, f"({dec})")

                cg.add_dysum(kc, kb, f"({dcb})")
                cg.add_dysum(kc, ke, f"({dce})")
                cg.add_dysum(kc, kc, f"({dcc})")

                cg.add_to_cur_code(cinit)
                cg.add_to_cur_code([f"res[{self.curr_index(comp.B)}] = -({cb})",
                                  f"res[{self.curr_index(comp.E)}] =  -({ce})",
                                  f"res[{self.curr_index(comp.C)}] =  -({cc})"])

            elif isinstance(comp, Capacitor):
                k = self.capa_index(comp)
                sn = self.state_index(comp)
                if transient:
                    cg.add_ysum(kp, f"sol[{k}]")
                    cg.add_ysum(kn, f"-(sol[{k}])")
                    cg.add_ysum(k, f"sol[{kp}] - sol[{kn}] - state_vec[{sn}]")

                    cg.add_dysum(kp, k, "1")
                    cg.add_dysum(kn, k, "-1")
                    cg.add_dysum(k, kp, "1")
                    cg.add_dysum(k, kn, "(-1)")
                    cg.add_to_cur_code([f"res[{self.curr_index(comp.p)}] = -(sol[{k}])",
                                        f"res[{self.curr_index(comp.n)}] = sol[{k}]"])
                else:
                    cg.add_ysum(k, f"sol[{k}]")
                    cg.add_dysum(k, k, "1")
                    cg.add_to_cur_code([f"res[{self.curr_index(comp.p)}] = sol[{k}]",
                                        f"res[{self.curr_index(comp.n)}] = -(sol[{k}])"])
            elif isinstance(comp, Inductor):
                k = self.induc_index(comp)
                sn = self.state_index(comp)
                if transient:
                    cg.add_ysum(kp, f"(-sol[{k}])")
                    cg.add_ysum(kn, f"sol[{k}]")
                    cg.add_ysum(k, f"sol[{k}] - state_vec[{sn}]")

                    cg.add_dysum(kp, k, "(-1)")
                    cg.add_dysum(kn, k, "1")
                    cg.add_dysum(k, k, "1")

                    cg.add_to_cur_code([f"res[{self.curr_index(comp.p)}] = state_vec[{sn}]",
                                        f"res[{self.curr_index(comp.n)}] = -state_vec[{sn}]"])
                else:
                    # sol[k] is the current through the inductor
                    # both nodes have the same voltage level
                    cg.add_ysum(k, f"sol[{kp}]- sol[{kn}]")
                    cg.add_ysum(kp, f"-sol[{k}]") # current leaves node
                    cg.add_ysum(kn, f"(sol[{k}])") # current enters node

                    cg.add_dysum(k, kp, "1")
                    cg.add_dysum(k, kn, "(-1)")
                    cg.add_dysum(kp, k, "(-1)")
                    cg.add_dysum(kn, k, "1")

                    cg.add_to_cur_code([f"res[{self.curr_index(comp.p)}] = sol[{k}]",
                                        f"res[{self.curr_index(comp.n)}] = -(sol[{k}])"])
            else:
                raise Exception("unknown component")

        for i in range(n):
            if i == 0: # skip current equation for ground, force voltage to be 0
                cg.add_to_y_code(["res[0]=sol[0]"])
                cg.add_to_dy_code(["res[0][0]=1"])
            else:
                cg.add_to_y_code([f"res[{i}]=" + " + ".join(cg.ysum[i])])
                for j in range(n):
                    if len(cg.dysum[i][j])>0:
                        cg.add_to_dy_code([f"res[{i}][{j}]=" + " + ".join(cg.dysum[i][j])])
        cg.add_to_y_code(["return res",""])
        cg.add_to_dy_code(["return res",""])
        cg.add_to_cur_code(["return res"])
        cg.add_to_init([""])
        l =  cg.init + cg.y_code + cg.dy_code + cg.cur_code
        code = "\n".join(l)
        #        print(code)
        d = {}
        exec(code,d)
        bla = d["bla"]
        computer = bla()
        return computer

    def _compute_state_vec(self, capa_voltages, induc_currents):
        state_vec = np.zeros(self.state_size())

        for comp in self.netw.components.values():
            if isinstance(comp, Capacitor):
                k = self.state_index(comp)
                if capa_voltages and  comp.name in capa_voltages:
                    state_vec[k] = capa_voltages[comp.name]
                else:
                    raise Exception(f"no voltage given for capacitor {comp.name}")
            if isinstance(comp, Inductor):
                k = self.state_index(comp)
                if induc_currents and comp.name in induc_currents:
                    state_vec[k] = induc_currents[comp.name]
                else:
                    raise Exception(f"no  current given for inductor {comp.name}")

        return state_vec

    def _compute_currents(self, bla, time, sol, state_vec):
        res = {}
        cv = bla.currents(time, sol,state_vec)
        for comp in self.netw.components.values():
            for p in comp.ports():
                res[p.pname()] = cv[self.curr_index(p)]
        return res


    def analyze(self,
                maxit=20,
                start_solution_vec=None,
                abstol= 1e-8,
                reltol= 1e-6,
                variables=None,
                capa_voltages=None,
                induc_currents=None,
                start_voltages=None,
                time =0,
                transient=False,
                compute_cond=False):
        if variables is None:
            variables = {}
        n = self._equation_size()
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

        if transient:
            state_vec = self._compute_state_vec(capa_voltages, induc_currents)
        else:
            state_vec = None

        c = self.generate_code(variables,transient)

        def f(x):
            return c.y(time, x, state_vec)

        def Df(x):
            return c.dy(time, x, state_vec)


        res = solving.solve(solution_vec, f, Df, abstol, reltol, maxit)
        if not isinstance(res, str):
            (sol, y, dfx, iterations, norm_y) = res
            if compute_cond:
                cond =  np.linalg.cond(dfx,'fro')
            else:
                cond=None
            norm_y = np.linalg.norm(y)
            currents = self._compute_currents(c,time, sol, state_vec)
            return Result(self.netw,
                          self,
                          iterations,
                          sol,
                          y,
                          norm_y,
                          cond,
                          currents)

        alfa = 0.5

        for i in range(20):
            print(("energy factor ",i))
            alfa = (alfa + 1) / 2
            res = solving.solve(solution_vec, f, Df, abstol, reltol, maxit,
                                x0=solution_vec0, alfa=alfa)
            if not isinstance(res, str):
                solution_vec = res[0]
                break
        if isinstance(res,str):
            print("failed getting initial solution")
            return res
        print(f"got initial solution, alfa={alfa}")

        while True:
            alfa = max(alfa / 1.1, 0)
            res = solving.solve(solution_vec, f, Df, abstol, reltol, maxit,
                                x0=solution_vec0, alfa=alfa)
            if isinstance(res, str):
                print(f"alfa={alfa}")
                return res
            if alfa <=0:
                break
            solution_vec = res[0]

        (sol, y, dfx, iterations, norm_y) = res
        norm_y = np.linalg.norm(y)
        if compute_cond:
            cond =  np.linalg.cond(dfx,'fro')
        else:
            cond=None
        currents = self._compute_currents(c, time, sol, state_vec)
        return Result(self.netw,
                      self,
                      iterations,
                      sol,
                      y,
                      norm_y,
                      cond,
                      currents)


    def solve_internal(self,
                       time,
                       maxit,
                       start_sol,
                       state_vec,
                       abstol,
                       reltol,
                       c,
                       compute_cond):

        def f(x):
            return c.y(time, x, state_vec)

        def Df(x):
            return c.dy(time, x, state_vec)

        res = solving.solve(start_sol, f, Df, abstol, reltol, maxit)
        if not isinstance(res, str):
            (sol, y, dfx, iterations, norm_y) = res
            if compute_cond:
                cond =  np.linalg.cond(dfx,'fro')
            else:
                cond=None
            currents = self._compute_currents(c, time, sol, state_vec)
            return Result(self.netw,
                          self,
                          iterations,
                          sol,
                          y,
                          norm_y,
                          cond,
                          currents)
        return res

    def transient(self,
                  maxtime,
                  timestep,
                  maxit=20,
                  start_solution_vec=None,
                  abstol= 1e-8,
                  reltol= 1e-6,
                  variables=None,
                  capa_voltages=None,
                  induc_currents=None,
                  start_voltages=None,
                  compute_cond=False):

        capa_voltages = capa_voltages or {}
        induc_currents = induc_currents or {}

        time = 0.0
        res = self.analyze(maxit=maxit,
                           start_solution_vec=start_solution_vec,
                           capa_voltages=capa_voltages,
                           induc_currents=induc_currents,
                           variables=variables,
                           start_voltages=start_voltages,
                           time=time,
                           abstol=abstol,
                           reltol=reltol,
                           transient=True,
                           compute_cond=compute_cond)
        if isinstance(res, str):
            raise Exception("can not find inital solution")

        state_vec = self._compute_state_vec(capa_voltages, induc_currents)

        solutions= []
        sol = res.solution_vec

        for comp in self.netw.components.values():
            if isinstance(comp, Capacitor):
                k = self.state_index(comp)
                if math.isnan(state_vec[k]):
                    v = res.get_voltage(comp.p) - res.get_voltage(comp.n)
                    state_vec[k] = v
            if isinstance(comp, Inductor):
                k = self.state_index(comp)
                if math.isnan(state_vec[k]):
                    curr = res.get_current(comp.p)
                    state_vec[k] = curr
        c = self.generate_code(variables,True)
        while time < maxtime:
            res = self.solve_internal(time,
                    maxit,
                    sol,
                    state_vec,
                    abstol,
                    reltol,
                    c,
                    compute_cond)
            if isinstance(res, str):
                raise Exception(f"fail at time {time}")

            solutions.append((time, res.get_voltages(), res.get_currents()))
            sol = res.solution_vec

            for comp in self.netw.components.values():
                if isinstance(comp, Capacitor):
                    capa = comp.get_capa(variables)
                    current = res.get_current(comp.p)
                    k = self.state_index(comp)
                    state_vec[k] += timestep*current/capa

                if isinstance(comp, Inductor):
                    indu = comp.induc
                    v = res.get_voltage(comp.p) - res.get_voltage(comp.n)
                    k = self.state_index(comp)
                    state_vec[k] += timestep*v/indu

            time += timestep
        return solutions


def pivot(res):
    time = []
    volts = {}
    currs = {}
    for (t,v,c) in res:
        time.append(t)
        for k in v:
            if not k in volts:
                volts[k] = []
            volts[k].append(v[k])
        for k in c:
            if not k in currs:
                currs[k] = []
            currs[k].append(c[k])
    for k in volts:
        volts[k] = np.array(volts[k])
    for k in currs:
        currs[k] = np.array(currs[k])

    return (np.array(time),volts,currs)

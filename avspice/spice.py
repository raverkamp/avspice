""" simple routines for experimenting with nodal analysis """

import collections
import math
import pprint as pp
import numbers
import numpy as np
from . import solving
from . import util

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
    def __init__(self, name):
        self.name = name

    def get_ports(self):
        """return the ports of this component"""
        raise NotImplementedError("method 'get_ports' is not implemented")

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

class Node2(Component):
    """a component with just two ports"""

    def __init__(self, name:str):
        super().__init__(name)

    def get_ports(self):
        return ["p", "n"]

class Resistor(Node2):
    """resistor"""
    def __init__(self, name, ohm):
        super().__init__(name)
        self._ohm = ohm

    def get_ohm(self, variables):
        return self.get_val(self._ohm, variables)

    def __repr__(self):
        return f"<Resistor {self.name}>"

class Current(Node2):
    """current source"""
    def __init__(self, name, amp):
        super().__init__(name)
        self.amp = amp

    def __repr__(self):
        return f"<Current {self.name}>"

    def get_amp(self, variables):
        return self.get_val(self.amp, variables)

    def code(self,cname):
        return ([], ([],f"{self.amp}"))

class Voltage(Node2):
    """voltage source"""
    def __init__(self, name:str, volts):
        super().__init__(name)
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

    def __init__(self, name:str, volts:float, frequency: float):
        super().__init__(name, volts)
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

    def __init__(self, name:str, volts:float, frequency: float):
        super().__init__(name, volts)
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
    def __init__(self, name, Is, Nut, lcut_off = -40, rcut_off=40):
        super().__init__(name)
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

    def __init__(self, name, capa):
        super().__init__(name)
        self._capa = capa

    def get_capa(self, variables):
        a = self.get_val(self._capa, variables)
        return a

    def __repr__(self):
        return f"<Capacitor {self.name}>"


class Inductor(Node2):
    """ a Spule" """

    def __init__(self, name, induc):
        super().__init__(name)
        self.induc = induc

    def get_induc(self, variables):
        return self.induc

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

    def __init__(self, name:str, IS:float, VT:float, beta_F:float, beta_R:float,
                 cutoff:float =40):
        super().__init__(name)
        self.IS = IS
        self.VT = VT
        self.beta_F = beta_F
        self.beta_R = beta_R

        self.lcutoff = -cutoff
        self.rcutoff = cutoff

    def get_ports(self):
        return ("B", "C", "E")

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

    def __init__(self, name:str, IS:float, VT:float, beta_F:float, beta_R:float,
                 cutoff:float=40):
        super().__init__(name)
        self.IS = IS
        self.VT = VT
        self.beta_F = beta_F
        self.beta_R = beta_R

        self.lcutoff = -cutoff
        self.rcutoff = cutoff

    def get_ports(self):
        return ("B", "C", "E")

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

Part =  collections.namedtuple("Part", ("name","component", "connections"))

class Network:
    """ this class describes the toplogy of an electrical network
        It only contains the topology"""

    def __init__(self):
        self.parts = []
        self.node_list = []
        self.part_dict = {}

    def add_component(self, name:str, comp: Component, nodes):
        assert isinstance(name, str), "name parameter must be a string"
        assert isinstance(comp, Component), "component parameter must be a component"
        ports = comp.get_ports()
        assert len(ports) == len(nodes)
        if name in self.part_dict:
            raise Exception(f"part with name {name} already exists")
        for node in nodes:
            assert isinstance(node, str)
            if not node in self.node_list:
                self.node_list.append(node)
        part = Part(name, comp, nodes)
        self.parts.append(part)
        self.part_dict[name] = part

    def addR(self, name, ohm, p, n):
        """add a curent source"""
        c = Resistor(name, ohm)
        self.add_component(name, c, (p, n))

    def addC(self, name, amp, p, n):
        """add a curent source"""
        c = Current(name, amp)
        self.add_component(name, c, (p, n))

    def addV(self, name, volts, p , n):
        v = Voltage(name, volts)
        self.add_component(name, v, (p,n))

    def addSineV(self, name, volts, frequency, p ,n):
        v = SineVoltage(name, volts, frequency)
        self.add_component(name, v, (p, n))

    def addSawV(self, name, volts, frequency, p, n):
        v = SawVoltage(name, volts, frequency)
        self.add_component(name, v, (p, n))

    def addD(self, name, Is, Nut, p, n):
        d = Diode(name, Is, Nut)
        self.add_component(name, d, (p, n))

    def addCapa(self, name, capa, p, n):
        c = Capacitor(name, capa)
        self.add_component(name, c, (p, n))

    def addInduc(self, name, induc, p, n):
        indu = Inductor(name, induc)
        self.add_component(name, indu, (p, n))

    def add_subcircuit(self, name, subcircuit, nodes):
        assert isinstance(subcircuit, SubCircuit)
        assert isinstance(name, str)
        assert util.is_str_seq(nodes)
        c = SubCircuitComponent(subcircuit)
        self.add_component(name,  c, nodes)


class Circuit(Network):
    def __init__(self):
        super().__init__()
        self.node_list.append("0")


class SubCircuit(Network):

    def __init__(self, export_nodes):
        super().__init__()
        assert util.is_str_seq(export_nodes), "nodes must be a sequence of strings"
        self.export_nodes = list(export_nodes)

class SubCircuitComponent(Component):

    def __init__(self, subcircuit):
        assert isinstance(subcircuit, SubCircuit)
        super().__init__("nix")
        self.subcircuit = subcircuit

    def get_ports(self):
        return self.subcircuit.export_nodes

class Result:
    """result of an analysis run"""
    def __init__(self, parts, analysis, iterations, solution_vec, y, y_norm, mat_cond, currents):
        assert isinstance(parts, list)
        self.analysis = analysis
        self.parts = parts
        self.solution_vec = solution_vec
        self.voltages = {}
        self.iterations = iterations
        self.mat_cond = mat_cond
        self.y = y
        self.y_norm = y_norm
        self.currents = currents

        for part in self.parts:
            ports = part.component.get_ports()
            nodes =  part.connections
            assert len(ports) == len(nodes)
            for i in range(len(ports)):
                port = ports[i]
                node = nodes[i]
                k = self.analysis.node_index(node)
                port_name = part.name +"."  + port
                self.voltages[port_name] = solution_vec[k]

    def get_voltages(self):
        return self.voltages

    def get_currents(self):
        return self.currents

    def __repr__(self):
        return repr({"voltages": self.voltages, "currents": self.currents})

    def get_voltage(self, portname):
        return self.voltages[portname]

    def get_current(self, port):
        return self.currents[port]

    def display(self):
        x = list(self.voltages.items())
        x.sort()
        print("--- Voltages ---")
        for (k,v) in x:
            print(k + " " + str(v))
        print(" --- currents ---")
        x = list(self.currents.items())
        x.sort()
        for (k,v) in x:
            print(k + " " + str(v))

class CodeGenerator:
    def __init__(self, n, n_curr_ports, transient):
        self.n = n
        self.n_curr_ports = n_curr_ports
        #        if transient:
        h_par = ", h"
        #       else:
        #          h_par = ""
        self.init = [
            "def bla():",
            "   return Computer()",
            "",
            "class Computer:",
            "    def __init__(self):",
            "        from avspice.ncomponents import NDiode, NNPNTransistor, NPNPTransistor,"
                    + "NVoltage, NSineVoltage, NSawVoltage"]
        self.y_code = [f"    def y(self, time, sol, state_vec{h_par}):",
                       "        import numpy as np",
                       f"        res = np.zeros({self.n})"]

        self.dy_code = [f"    def dy(self, time, sol,state_vec{h_par}):",
                        "        import numpy as np",
                        f"        res = np.zeros(({self.n},{self.n}))"]
        self.cur_code  = ["    def currents(self, time, sol, state_vec):",
                          "        import numpy as np",
                          f"        res = np.zeros({self.n_curr_ports})"]
        self.ysum=[]
        for _ in range(self.n):
            self.ysum.append([])
        self.dysum=[]
        for _ in range(self.n):
            x = []
            for _ in range(n):
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

    def __init__(self, circuit):
        assert isinstance(circuit, Circuit), "need a circuit"

        parts = []
        node_list = ["0"]
        # the nodes are already absolute!
        def add_sparts(prefix, sc, nodes):
            if isinstance(sc, SubCircuit):
                export_nodes = sc.export_nodes
            else:
                export_nodes = []
            for part in sc.parts:
                newcons = []
                for con in part.connections:
                    i = export_nodes.index(con) if con in export_nodes else -1
                    if  i>=0:
                        x = nodes[i]
                    else:
                        x = prefix + con
                    newcons.append(x)
                    if not x in node_list:
                        node_list.append(x)

                if isinstance(part.component, SubCircuitComponent):
                    sc = part.component.subcircuit
                    newprefix = prefix + part.name + "/"
                    add_sparts(newprefix, sc, newcons)
                else:
                    component = part.component
                    newname = prefix + part.name
                    newpart = Part(newname, component, newcons)
                    parts.append(newpart)

        add_sparts("", circuit, [])

        self.parts = parts
        self.node_list = node_list

        self.voltage_list = []
        self.capa_list = []
        self.induc_list  = []
        self.curr_port_list = []

        for part in self.parts:
            if isinstance(part.component, Voltage):
                self.voltage_list.append(part)
            if isinstance(part.component, Capacitor):
                self.capa_list.append(part)
            if isinstance(part.component, Inductor):
                self.induc_list.append(part)
            for port in part.component.get_ports():
                self.curr_port_list.append((part.name, port))
        self.port_node_indexes = {}
        for node in self.node_list:
            self.port_node_indexes[node] =  self.node_index(node)
        for part in self.parts:
            for (port, node) in zip(part.component.get_ports(), part.connections):
                self.port_node_indexes[f"{part.name}.{port}"] = self.node_index(node)

    def node_index(self, node):
        return self.node_list.index(node)

    def voltage_index(self, voltage):
        return self.voltage_list.index(voltage) + len(self.node_list)

    def capa_index(self, capa):
        return self.capa_list.index(capa) + len(self.node_list) + len(self.voltage_list)

    def induc_index(self, induc):
        return (self.induc_list.index(induc)
                + len(self.node_list)
                + len(self.voltage_list)
                + len(self.capa_list))

    def state_index(self, part):
        if isinstance(part.component, Capacitor):
            return self.capa_list.index(part)
        if isinstance(part.component, Inductor):
            return self.induc_list.index(part) + len(self.capa_list)
        raise Exception(f"no state index for component {part}")

    def state_index_y(self, part):
        return (len(self.node_list)
                + len(self.voltage_list)
                + len(self.capa_list)
                + len(self.induc_list)
                + self.state_index(part))

    def state_size(self):
        return len(self.capa_list) + len(self.induc_list)


    def _equation_size(self, transient):
        return (len(self.node_list)
                + len(self.voltage_list)
                + len(self.capa_list)
                + len(self.induc_list)
                + (len(self.capa_list) + len(self.induc_list) if transient else 0))

    def curr_index(self, partname, port):
        return self.curr_port_list.index((partname, port))

    def generate_code(self, variables, transient):
        n = self._equation_size(transient)
        cg = CodeGenerator(n, len(self.curr_port_list), transient)

        counter = 0
        for part in self.parts:
            comp = part.component
            cname = part.name.replace("/","_") + str(counter)
            nodes = part.connections
            if isinstance(comp, Node2):
                kp = self.node_index(nodes[0])
                kn = self.node_index(nodes[1])
                curr_index_p  = self.curr_index(part.name, "p")
                curr_index_n  = self.curr_index(part.name, "n")

            if isinstance(comp, Voltage):
                (vinit, (pre , expr)) = comp.code(cname, variables)
                cg.add_to_init(vinit)
                k = self.voltage_index(part)
                cg.add_to_y_code(pre)
                cg.add_ysum(kp, f"sol[{k}]")
                cg.add_ysum(kn, f"(-sol[{k}])")
                cg.add_ysum(k,f"(sol[{kp}] - sol[{kn}]) - ({expr})")
                cg.add_dysum(kp, k, "1")
                cg.add_dysum(kn, k, "-1")
                cg.add_dysum(k, kp, "1")
                cg.add_dysum(k, kn, "-1")

                cg.add_to_cur_code([f"res[{curr_index_p}] = sol[{k}]",
                                    f"res[{curr_index_n}] = -(sol[{k}])"])

            elif isinstance(comp, Current):
                (cinit, (pre, expr)) = comp.code(cname)
                cg.add_to_init(cinit)
                cg.add_to_y_code(pre)
                cg.add_ysum(kp, f"({expr})")
                cg.add_ysum(kn, f"(-({expr}))")

                cg.add_to_cur_code(pre)
                cg.add_to_cur_code([f"res[{curr_index_p}] = -({expr})",
                                 f"res[{curr_index_n}] = {expr}"])

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
                                    f"res[{curr_index_p}] = ({name})",
                                    f"res[{curr_index_n}] =  -({name})"])

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
                cg.add_to_cur_code([f"res[{curr_index_p}] = {curr}",
                                  f"res[{curr_index_n}] =  -({curr})"])

            elif isinstance(comp, (NPNTransistor, PNPTransistor)):
                kb = self.node_index(nodes[0])
                ke = self.node_index(nodes[2])
                kc = self.node_index(nodes[1])

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

                curr_index_B  = self.curr_index(part.name, "B")
                curr_index_E  = self.curr_index(part.name, "E")
                curr_index_C  = self.curr_index(part.name, "C")

                cg.add_to_cur_code([f"res[{curr_index_B}] = -({cb})",
                                  f"res[{curr_index_E}] =  -({ce})",
                                  f"res[{curr_index_C}] =  -({cc})"])

            elif isinstance(comp, Capacitor):
                k = self.capa_index(part)

                if transient:
                    sn = self.state_index(part)
                    sny = self.state_index_y(part)
                    # sol[k] current through capacitor
                    # sol[sny] voltage accross capacitor
                    # state_vec(sn) voltage of last iteration
                    capa = comp.get_capa(variables)
                    cg.add_ysum(kp, f"sol[{k}]")
                    cg.add_ysum(kn, f"-(sol[{k}])")
                    cg.add_ysum(k, f"sol[{kp}] - sol[{kn}] - sol[{sny}]")
                    #cg.add_ysum(k, f"sol[{kp}] - sol[{kn}] - state_vec[{sn}]")

                    cg.add_ysum(sny, f"sol[{sny}] + h * sol[{k}]/{capa} - state_vec[{sn}]")

                    cg.add_dysum(kp, k, "1")
                    cg.add_dysum(kn, k, "-1")
                    cg.add_dysum(k, kp, "1")
                    cg.add_dysum(k, kn, "(-1)")
                    cg.add_dysum(k, sny, "(-1)")
                    cg.add_dysum(sny, sny, "1")
                    cg.add_dysum(sny, k,f"+h/{capa}")

                    cg.add_to_cur_code([f"res[{curr_index_p}] = -(sol[{k}])",
                                        f"res[{curr_index_n}] = sol[{k}]"])
                else:
                    cg.add_ysum(k, f"sol[{k}]")
                    cg.add_dysum(k, k, "1")
                    cg.add_to_cur_code([f"res[{curr_index_p}] = sol[{k}]",
                                        f"res[{curr_index_n}] = -(sol[{k}])"])
            elif isinstance(comp, Inductor):
                k = self.induc_index(part)
                if transient:
                    sn = self.state_index(part)
                    sny = self.state_index_y(part)
                    induc = comp.get_induc(variables)
                    # sol[k] current through inductor
                    # sol[sny] also current through conductor
                    # state_vec(sn) current through conductor of last ieteration
                    cg.add_ysum(kp, f"(-sol[{k}])")
                    cg.add_ysum(kn, f"sol[{k}]")
                    cg.add_ysum(k, f"sol[{k}] - sol[{sny}]")
                    cg.add_ysum(sny, f"sol[{sny}] - h *  (sol[{kp}] "
                                + f" - sol[{kn}])/{induc} - state_vec[{sn}]")

                    cg.add_dysum(kp, k, "(-1)")
                    cg.add_dysum(kn, k, "1")
                    cg.add_dysum(k, k, "1")
                    cg.add_dysum(k, sny, "-1")
                    cg.add_dysum(sny,  sny, "1")
                    cg.add_dysum(sny, kp, f"-h /{induc}")
                    cg.add_dysum(sny, kn, f"h /{induc}")



                    cg.add_to_cur_code([f"res[{curr_index_p}] = state_vec[{sn}]",
                                        f"res[{curr_index_n}] = -state_vec[{sn}]"])
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

                    cg.add_to_cur_code([f"res[{curr_index_p}] = sol[{k}]",
                                        f"res[{curr_index_n}] = -(sol[{k}])"])
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

        for part in self.parts:
            if isinstance(part.component, Capacitor):
                k = self.state_index(part)
                if capa_voltages and  part.name in capa_voltages:
                    state_vec[k] = capa_voltages[part.name]
                else:
                    raise Exception(f"no voltage given for capacitor {part.name}")
            if isinstance(part.component, Inductor):
                k = self.state_index(part)
                if induc_currents and part.name in induc_currents:
                    state_vec[k] = induc_currents[part.name]
                else:
                    raise Exception(f"no  current given for inductor {part.name}")

        return state_vec

    def _compute_currents(self, bla, time, sol, state_vec):
        res = {}
        cv = bla.currents(time, sol,state_vec)
        for  (name, port) in self.curr_port_list:
            res[name + "." +port] = cv[self.curr_index(name, port)]
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
        n = self._equation_size(transient)
        if start_solution_vec is None:
            if start_voltages is None:
                solution_vec0 = np.zeros(n)
            else:
                solution_vec0 = np.zeros(n)
                for vk in start_voltages:
                    n = self.port_node_indexes[vk]
                    solution_vec0[n] = start_voltages[vk]
        else:
            solution_vec0 = start_solution_vec

        solution_vec = solution_vec0

        if transient:
            state_vec = self._compute_state_vec(capa_voltages, induc_currents)
        else:
            state_vec = None

        computer = self.generate_code(variables,transient)

        def f(x):
            return computer.y(time, x, state_vec, 0)

        def Df(x):
            return computer.dy(time, x, state_vec, 0)

        res = solving.solve(solution_vec, f, Df, abstol, reltol, maxit)
        if not isinstance(res, str):
            (sol, y, dfx, iterations, norm_y) = res
            if compute_cond:
                cond =  np.linalg.cond(dfx,'fro')
            else:
                cond=None
            norm_y = np.linalg.norm(y)
            currents = self._compute_currents(computer, time, sol, state_vec)
            return Result(self.parts,
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
        currents = self._compute_currents(computer, time, sol, state_vec)
        return Result(self.parts,
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
                       compute_cond,
                       h):

        def f(x):
            return c.y(time, x, state_vec, h)

        def Df(x):
            return c.dy(time, x, state_vec, h)

        res = solving.solve(start_sol, f, Df, abstol, reltol, maxit)
        if not isinstance(res, str):
            (sol, y, dfx, iterations, norm_y) = res
            if compute_cond:
                cond =  np.linalg.cond(dfx,'fro')
            else:
                cond=None
            currents = self._compute_currents(c, time, sol, state_vec)
            return Result(self.parts,
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
        max_timestep =  timestep

        min_timestep = timestep / 10000.0
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
        computer = self.generate_code(variables,True)
        while time < maxtime:
            res = self.solve_internal(time,
                    maxit,
                    sol,
                    state_vec,
                    abstol,
                    reltol,
                    computer,
                    compute_cond,
                    timestep)
            if isinstance(res, str):
                timestep = timestep / 2
                if timestep < min_timestep:
                    print(f"fail at time {time}: {res}, stepisze={timestep}")
                    return solutions
                print("dec step", time, timestep)
                continue
            a = timestep * 1.05
            if a < max_timestep:
                timestep = a
                print("inc step", time, timestep)
            solutions.append((time, res.get_voltages(), res.get_currents()))
            sol = res.solution_vec

            for part in self.parts:
                comp =  part.component
                if isinstance(comp, (Capacitor, Inductor)):
                    k = self.state_index(part)
                    state_vec[k] = sol[self.state_index_y(part)]

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

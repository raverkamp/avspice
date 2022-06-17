""" simple routines for experimenting with nodal analysis """

import collections
import math
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
                    raise Exception(f"did not find value for variable '{var_or_num.name}' in '{self.name}'")
                return var_or_num.default
            return val
        raise Exception("bug")

class Node2(Component):
    """a component with just two ports"""

    def __init__(self, name: str):
        super().__init__(name)

    def get_ports(self):
        return ["p", "n"]

class Resistor(Node2):
    """resistor"""
    def __init__(self, name, ohm):
        super().__init__(name)
        self._ohm = ohm

    def get_resistance(self):
        return self._ohm

    def __repr__(self):
        return f"<Resistor {self._ohm}>"

class Current(Node2):
    """current source"""
    def __init__(self, name, amp):
        super().__init__(name)
        self.amp = amp

    def __repr__(self):
        return f"<Current {self.name}>"

    def code(self, cg, cname):
        x = cg.get_value_code(self.amp)
        return ([], ([],f"{x}"))

class Voltage(Node2):
    """voltage source"""
    def __init__(self, name:str, volts):
        super().__init__(name)
        assert isinstance(volts, (numbers.Number, Variable)), "volts must be a variable or a number"
        self._volts = volts

    def __repr__(self):
        return f"<Voltage {self._volts}>"

    def code(self, cg, cname):
        v = cg.get_value_code(self._volts)
        init = [f"self.{cname} = NVoltage({v})"]
        voltage = ([f"{cname}_voltage = self.{cname}.voltage(time)"],f"{cname}_voltage")
        return (init, voltage)

class SineVoltage(Voltage):

    def __init__(self, name:str, volts:float, frequency: float):
        super().__init__(name, volts)
        self._frequency = frequency

    def code(self, cg, cname):
        v = cg.get_value_code(self._volts)
        f = cg.get_value_code(self._frequency)
        init = [f"self.{cname} = NSineVoltage({v}, {f})"]
        voltage = ([f"{cname}_voltage = self.{cname}.voltage(time)"],f"{cname}_voltage")
        return (init, voltage)

class SawVoltage(Voltage):

    def __init__(self, name:str, volts:float, frequency: float):
        super().__init__(name, volts)
        self._frequency = frequency

    def code(self, cg, cname):
        v = cg.get_value_code(self._volts)
        init = [f"self.{cname} = NSawVoltage({v}, {self._frequency})"]
        voltage = ([f"{cname}_voltage = self.{cname}.voltage(time)"],f"{cname}_voltage")
        return (init, voltage)

class PieceWiseLinearVoltage(Voltage):

    def __init__(self, name:str, pairs):
        super().__init__(name,0)
        self.pairs = list(pairs)

    def code(self, cg, cname):
        a = list(self.pairs)
        a.sort(key=lambda x: x[0])
        vx  = list([x for (x,y) in a])

        vy  = "[" + ", ".join(list([cg.get_value_code(y) for (x,y) in a])) + "]"

        init = [f"self.{cname} = NPieceWiseLinearVoltage({vx},  {vy})"]
        voltage = ([f"{cname}_voltage = self.{cname}.voltage(time)"],f"{cname}_voltage")
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

    def get_capacitance(self):
        return self._capa

    def __repr__(self):
        return f"<Capacitor {self.name}>"


class Inductor(Node2):
    """ a Spule" """

    def __init__(self, name, induc):
        super().__init__(name)
        self.induc = induc

    def get_inductance(self):
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

    def __repr__(self):
        return f"<Transistor {self.name}>"

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

class TransientResult:
    """result of a transient simulation"""

    def __init__(self, time_points, voltages, currents):
        if len(time_points) == 0:
            raise Exception("init has length 0")
        volts = {}
        currs = {}
        self.start_time = time_points[0]
        self.end_time = time_points[len(time_points)-1]
        if len(time_points) != len(voltages) or len(time_points) != len(currents):
            raise Exception("length of lists do not match")
        for k in voltages[0]:
            volts[k] = []
        for k in currents[0]:
            currs[k] = []
        for i in range(len(time_points)):
            v = voltages[i]
            c = currents[i]
            if len(v)  != len(volts):
                raise Exception("length for volts does not match")
            if len(c)  != len(currs):
                raise Exception("length for currents does not match")
            for (k, val) in volts.items():
                val.append(v[k])
            for (k, val) in currs.items():
                val.append(c[k])

        self.voltages = {}
        self.currents = {}
        self.time = np.array(time_points)

        for (k, val) in volts.items():
            self.voltages[k] = np.array(val)

        for (k, val) in currs.items():
            self.currents[k] = np.array(val)

    def get_voltage(self, k):
        return self.voltages[k]

    def get_current(self, k):
        return self.currents[k]

    def get_time(self):
        return self.time

    def get_voltage_at(self, t, k):
        return util.linear_interpolate(self.time, self.get_voltage(k), t)

    def get_current_at(self, t, k):
        return util.linear_interpolate(self.time, self.get_current(k), t)



class Result:
    """result of an analysis run"""
    def __init__(self, parts, analysis, iterations, solution_vec, y, y_norm, mat_cond, currents):
        assert isinstance(parts, list)
        self.solution_vec = solution_vec
        self.voltages = {}
        self.iterations = iterations
        self.mat_cond = mat_cond
        self.y = y
        self.y_norm = y_norm
        self.currents = currents

        for part in parts:
            ports = part.component.get_ports()
            nodes =  part.connections
            assert len(ports) == len(nodes)

            for (port,node) in zip(ports, nodes):
                k = analysis.node_index(node)
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
        self.component_init = []
        self.init = [
            "def bla(variables):",
            "   return Computer(variables)",
            "",
            "class Computer:",
            "    def __init__(self, variables):",
            "        from avspice.ncomponents import NDiode, NNPNTransistor, NPNPTransistor,"
                    + "NVoltage, NSineVoltage, NSawVoltage, NPieceWiseLinearVoltage"]
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
        self.variables = {}

    def get_var_code(self, variable):
        assert isinstance(variable, Variable), "variable parameter must be instance of variable"
        if variable in self.variables:
            k = self.variables[variable]
        else:
            k = len(self.variables)
            self.variables[variable] = k
        return f"self.variables[{k}]"

    def get_value_code(self, x):
        if  isinstance(x, Variable):
            return self.get_var_code(x)
        elif isinstance(x, (float, int)):
            return str(x)
        else:
            raise Exception("wrong type for scalar value:" + str(x))

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

    def add_to_cinit(self,l):
        self.add_to_method(self.component_init, l)

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
        d={}
        for n in self.node_list:
            d[n] = 0
        for p in parts:
            for c in p.connections:
                d[c] =d[c] +1
        for (k,v) in d.items():
            if v==0:
                raise Exception(f"no part conencted to '{k}")
            if v==1:
                raise Exception(f"only one connection to '{k}")



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

    def generate_code(self, transient):
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
                (vinit, (pre , expr)) = comp.code(cg, cname)
                cg.add_to_cinit(vinit)
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
                (cinit, (pre, expr)) = comp.code(cg, cname)
                cg.add_to_cinit(cinit)
                cg.add_to_y_code(pre)
                cg.add_ysum(kp, f"({expr})")
                cg.add_ysum(kn, f"(-({expr}))")

                cg.add_to_cur_code(pre)
                cg.add_to_cur_code([f"res[{curr_index_p}] = -({expr})",
                                 f"res[{curr_index_n}] = {expr}"])

            elif isinstance(comp, Resistor):
                r = cg.get_value_code(comp.get_resistance())
                G = f"self.{cname}_G"
                cg.add_to_init([f"{G}=1/{r}"])
                name = f"current_{cname}"
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
                cg.add_to_cinit(init_d)
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

                cg.add_to_cinit(init_t)
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
                    capa1 = comp.get_capacitance()
                    capa = cg.get_value_code(capa1)
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
                    induc1 = comp.get_inductance()
                    induc = cg.get_value_code(induc1)
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

        cg.add_to_init([f"self.variables = [0] * {len(cg.variables)}"])
        vard = {}
        for (var, k) in cg.variables.items():
            vard[var.name] = k
            cg.add_to_init([f"self.variables[{k}] = {var.default}"])
        cg.add_to_init([f"self.variable_map= {repr(vard)}",
                        "for (k,v) in variables.items():",
                        "     self.set_variable(k,v)"])


        x = ["    def set_variable(self, name, value):",
             "        if not name in self.variable_map:",
             "            raise Exception('unknown variable')",
             "        self.variables[self.variable_map[name]] = value",
             ""]


        l =  cg.init + cg.component_init + [""] + x + cg.y_code + cg.dy_code + cg.cur_code
        code = "\n".join(l)
        d = {}
        #        print(code)
        exec(code,d)
        bla = d["bla"]
        return bla

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

    def _compute_voltages(self, parts, solution_vec):
        voltages = {}
        for part in parts:
            ports = part.component.get_ports()
            nodes = part.connections
            assert len(ports) == len(nodes)
            for (port, node) in zip(ports, nodes):
                k = self.node_index(node)
                port_name = part.name +"."  + port
                voltages[port_name] = solution_vec[k]
        return voltages

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

        bla = self.generate_code(transient)
        computer = bla(variables)
        for (k,v) in variables.items():
            computer.set_variable(k,v)

        def f(x):
            return computer.y(time, x, state_vec, 0)

        def Df(x):
            return computer.dy(time, x, state_vec, 0)

        res = solving.solve_alfa(solution_vec, f, Df, abstol, reltol, maxit)
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
        else:
            return res


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
        if isinstance(res, str):
            return res
        if compute_cond:
            cond =  np.linalg.cond(res.dfx,'fro')
        else:
            cond=None
        currents = self._compute_currents(c, time, res.x, state_vec)
        voltages = self._compute_voltages(self.parts, res.x)
        return ((res.x, res.y, res.norm_y, cond, res.iterations), voltages, currents)


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

        voltage_list = []
        current_list = []
        time_list = []

        sol = res.solution_vec
        bla = self.generate_code(True)
        computer = bla(variables or {})
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
                print("res: " + res)
                timestep = timestep / 2
                if timestep < min_timestep * 2:
                    n = self._equation_size(True)
                    #                    sol = np.zeros(n) + np.random.rand(n)
                    sol = (np.random.rand(n)*10)
                    timestep  = timestep * 4
                if timestep < min_timestep:
                    print(f"fail at time {time}: {res}, stepisze={timestep}")
                    break
                print("dec step", time, timestep)
                continue
            a = timestep * 1.05
            if a < max_timestep:
                timestep = a
                print("inc step", time, timestep)
            ((sol_x, sol_y, sol_ny, sol_cond, sol_iterations), voltages, currents) = res
            time_list.append(time)
            voltage_list.append(voltages)
            current_list.append(currents)

            sol = sol_x

            for part in self.parts:
                comp =  part.component
                if isinstance(comp, (Capacitor, Inductor)):
                    k = self.state_index(part)
                    state_vec[k] = sol[self.state_index_y(part)]

            time += timestep
        return TransientResult(time_list,  voltage_list, current_list)


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

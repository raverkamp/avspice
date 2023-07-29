""" simple routines for experimenting with nodal analysis """

import pprint as pp
from typing import TypeAlias, Union, Optional, Callable, Any
import numpy as np
import numpy.typing as npt

from .circuits import (
    Voltage,
    Node2,
    Node2Current,
    Circuit,
    SubCircuit,
    Inductor,
    Capacitor,
    SubCircuitComponent,
    Part,
    NPort,
    Network,
    VoltageControlledVoltageSource,
    LinearVoltageControlledVoltageSource,
    VoltageControlledCurrentSource,
)
from .variable import Variable
from . import solving
from . import util


np_float_vec: TypeAlias = npt.NDArray[np.float64]
float_vec: TypeAlias = Union[list[float], np_float_vec]


class TransientResult:
    """result of a transient simulation"""

    def __init__(
        self,
        time_points: list[float],
        voltages: list[dict[str, float]],
        currents: list[dict[str, float]],
    ) -> None:
        if len(time_points) == 0:
            raise ValueError("init has length 0")
        volts: dict[str, list[float]] = {}
        currs: dict[str, list[float]] = {}
        self.start_time = time_points[0]
        self.end_time = time_points[len(time_points) - 1]
        if len(time_points) != len(voltages) or len(time_points) != len(currents):
            raise ValueError("length of lists do not match")
        for k in voltages[0]:
            volts[k] = []
        for k in currents[0]:
            currs[k] = []
        for i in range(len(time_points)):
            v = voltages[i]
            c = currents[i]
            if len(v) != len(volts):
                raise ValueError("length for volts does not match")
            if len(c) != len(currs):
                raise ValueError("length for currents does not match")
            for k, val in volts.items():
                val.append(v[k])
            for k, val in currs.items():
                val.append(c[k])

        self.voltages: dict[str, np_float_vec] = {}
        self.currents: dict[str, np_float_vec] = {}
        self.time = np.array(time_points)

        for k, val in volts.items():
            self.voltages[k] = np.array(val)

        for k, val in currs.items():
            self.currents[k] = np.array(val)

    def get_voltage(self, k: str) -> np_float_vec:
        return self.voltages[k]

    def get_current(self, k: str) -> np_float_vec:
        return self.currents[k]

    def get_time(self) -> np_float_vec:
        return self.time

    def get_voltage_at(self, t: float, k: str) -> float:
        return util.linear_interpolate(self.time, self.get_voltage(k), t)

    def get_current_at(self, t: float, k: str) -> float:
        return util.linear_interpolate(self.time, self.get_current(k), t)


class Result:
    """result of an analysis run"""

    def __init__(
        self,
        parts: list[Part],
        analysis: "Analysis",
        iterations: int,
        solution_vec: np_float_vec,
        y: np_float_vec,
        y_norm: float,
        mat_cond: float,
        currents: dict[str, float],
    ):
        assert isinstance(parts, list)
        self.solution_vec = solution_vec
        self.voltages: dict[str, float] = {}
        self.iterations = iterations
        self.mat_cond = mat_cond
        self.y = y
        self.y_norm = y_norm
        self.currents = currents

        for node in analysis.node_list:
            k = analysis.node_index(node)
            self.voltages[node] = solution_vec[k]

        for part in parts:
            ports = part.component.get_ports()
            nodes = part.connections
            assert len(ports) == len(nodes)

            for port, node in zip(ports, nodes):
                k = analysis.node_index(node)
                port_name = part.name + "." + port
                self.voltages[port_name] = solution_vec[k]
        for node in analysis.node_list:
            k = analysis.node_index(node)
            self.voltages[node] = solution_vec[k]

    def get_voltages(self) -> dict[str, float]:
        return self.voltages

    def get_currents(self) -> dict[str, float]:
        return self.currents

    def __repr__(self) -> str:
        return repr({"voltages": self.voltages, "currents": self.currents})

    def get_voltage(self, portname: str) -> float:
        return self.voltages[portname]

    def get_current(self, port: str) -> float:
        return self.currents[port]

    def display(self) -> None:
        x = list(self.voltages.items())
        x.sort()
        print("--- Voltages ---")
        for k, v in x:
            print(k + " " + str(v))
        print(" --- currents ---")
        x = list(self.currents.items())
        x.sort()
        for k, v in x:
            print(k + " " + str(v))


class CodeGenerator:
    """utility class for code generation"""

    def __init__(self, n: int, n_curr_ports: int) -> None:
        self.n = n
        self.n_curr_ports = n_curr_ports
        h_par = ", h"
        self.component_init: list[str] = []
        self.init = [
            "def bla(variables):",
            "   return Computer(variables)",
            "",
            "class Computer(ComputerBase):",
            "    def __init__(self, variables):",
            "        from avspice.ncomponents import NDiode, NZDiode, NNPNTransistor,"
            + " NPNPTransistor, NFET, NJFETn,"
            + "NVoltage, NSineVoltage, NSawVoltage, NPwmVoltage, NPieceWiseLinearVoltage,"
            + " NPeriodicPieceWiseLinearVoltage, NLinearVoltageControlledVoltageSource,"
            + " NLinearVoltageControlledCurrentSource",
        ]
        self.y_code = [
            f"    def y(self, time, sol, state_vec{h_par}):",
            "        import numpy as np",
            f"        res = np.zeros({self.n})",
        ]

        self.dy_code = [
            f"    def dy(self, time, sol,state_vec{h_par}):",
            "        import numpy as np",
            f"        res = np.zeros(({self.n},{self.n}))",
        ]
        self.cur_code = [
            "    def currents(self, time, sol, state_vec):",
            "        import numpy as np",
            f"        res = np.zeros({self.n_curr_ports})",
        ]
        self.ysum: list[list[str]] = []
        for _ in range(self.n):
            self.ysum.append([])
        self.dysum: list[list[list[str]]] = []
        for _ in range(self.n):
            x: list[list[str]] = []
            for _ in range(n):
                x.append([])
            self.dysum.append(x)
        self.variables: dict[Variable, int] = {}

    def get_var_code(self, variable: Variable) -> str:
        assert isinstance(
            variable, Variable
        ), "variable parameter must be instance of variable"
        if variable in self.variables:
            k = self.variables[variable]
        else:
            k = len(self.variables)
            self.variables[variable] = k
        return f"self.variables[{k}]"

    def get_value_code(self, x: Union[Variable, float]) -> str:
        if isinstance(x, Variable):
            return self.get_var_code(x)
        elif isinstance(x, (float, int)):
            return str(x)
        else:
            raise ValueError("wrong type for scalar value:" + str(x))

    def add_ysum(self, k: int, e: str) -> None:
        self.ysum[k].append(e)

    def add_dysum(self, k: int, j: int, e: str) -> None:
        self.dysum[k][j].append(e)

    def add_to_method(self, m: list[str], lines_or_str: Union[str, list[str]]) -> None:
        if isinstance(lines_or_str, str):
            l = [lines_or_str]
        else:
            l = lines_or_str
        for x in l:
            assert isinstance(x, str)
            m.append("        " + x)

    def add_to_cinit(self, l: list[str]) -> None:
        self.add_to_method(self.component_init, l)

    def add_to_init(self, l: list[str]) -> None:
        self.add_to_method(self.init, l)

    def add_to_y_code(self, l: list[str]) -> None:
        self.add_to_method(self.y_code, l)

    def add_to_dy_code(self, l: list[str]) -> None:
        self.add_to_method(self.dy_code, l)

    def add_to_cur_code(self, l: list[str]) -> None:
        self.add_to_method(self.cur_code, l)


class ComputerBase:
    def y(
        self, time: float, sol: np_float_vec, vec: np_float_vec, h: float
    ) -> np_float_vec:
        raise NotImplementedError()

    def dy(
        self, time: float, sol: np_float_vec, vec: np_float_vec, h: float
    ) -> np_float_vec:
        raise NotImplementedError()

    def currents(
        self, time: float, sol: np_float_vec, vec: np_float_vec
    ) -> np_float_vec:
        raise NotImplementedError()


def flattenCircuit(circuit: Circuit) -> tuple[list[Part], list[str]]:
    # the circuit which is might contain subcrcuits is flattened into one
    # big circuit without subcircuits
    # name of parts in subcircuits is prefixed
    # the same for the nodes in subcicuits

    parts = []
    node_list = ["0"]

    def add_subcircuit_parts(prefix: str, sc: Network, nodes: list[str]) -> None:
        # prefix is prefixed to name of nodes and parts
        # sc is the circuit or subcircuit
        # nodes the nodes the subcircuit is connected to
        #    the nodes are already absolute!
        if isinstance(sc, SubCircuit):
            export_nodes = sc.export_nodes
        else:
            export_nodes = []

        for part in sc.parts:
            new_connections = []
            for con in part.connections:
                i = export_nodes.index(con) if con in export_nodes else -1
                if i >= 0:
                    x = nodes[i]
                else:
                    x = prefix + con
                new_connections.append(x)
                if not x in node_list:
                    node_list.append(x)

            if isinstance(part.component, SubCircuitComponent):
                sc = part.component.subcircuit
                newprefix = prefix + part.name + "/"
                add_subcircuit_parts(newprefix, sc, new_connections)
            else:
                component = part.component
                newname = prefix + part.name
                newpart = Part(newname, component, new_connections)
                parts.append(newpart)

    add_subcircuit_parts("", circuit, [])
    return (parts, node_list)


class Analysis:
    """captures all data for analysis"""

    def __init__(self, circuit: Circuit) -> None:
        assert isinstance(circuit, Circuit), "need a circuit"

        (self.parts, self.node_list) = flattenCircuit(circuit)

        self.voltage_list: list[Part] = []
        self.capa_list: list[Part] = []
        self.induc_list: list[Part] = []
        self.curr_port_list: list[tuple[str, str]] = []

        for part in self.parts:
            if isinstance(part.component, (Voltage, VoltageControlledVoltageSource)):
                self.voltage_list.append(part)
            if isinstance(part.component, Capacitor):
                self.capa_list.append(part)
            if isinstance(part.component, Inductor):
                self.induc_list.append(part)
            for port in part.component.get_ports():
                self.curr_port_list.append((part.name, port))
        self.port_node_indexes = {}
        for node in self.node_list:
            self.port_node_indexes[node] = self.node_index(node)
        for part in self.parts:
            for port, node in zip(part.component.get_ports(), part.connections):
                self.port_node_indexes[f"{part.name}.{port}"] = self.node_index(node)
        d = {}
        for n in self.node_list:
            d[n] = 0
        for p in self.parts:
            for c in p.connections:
                d[c] = d[c] + 1
        for k, v in d.items():
            if v == 0:
                raise Exception(f"no part conencted to '{k}")
            if v == 1:
                raise Exception(f"only one connection to '{k}")

    def node_index(self, node: str) -> int:
        return self.node_list.index(node)

    def port_node_index(self, part: Part, port: str) -> int:
        component = part.component
        x = component.get_ports().index(port)
        node = part.connections[x]
        return self.node_index(node)

    def voltage_index(self, voltage: Part) -> int:
        return self.voltage_list.index(voltage) + len(self.node_list)

    def capa_index(self, capa: Part) -> int:
        return self.capa_list.index(capa) + len(self.node_list) + len(self.voltage_list)

    def induc_index(self, induc: Part) -> int:
        return (
            self.induc_list.index(induc)
            + len(self.node_list)
            + len(self.voltage_list)
            + len(self.capa_list)
        )

    def state_index(self, part: Part) -> int:
        if isinstance(part.component, Capacitor):
            return self.capa_list.index(part)
        if isinstance(part.component, Inductor):
            return self.induc_list.index(part) + len(self.capa_list)
        raise Exception(f"no state index for component {part}")

    def state_index_y(self, part: Part) -> int:
        return (
            len(self.node_list)
            + len(self.voltage_list)
            + len(self.capa_list)
            + len(self.induc_list)
            + self.state_index(part)
        )

    def state_size(self) -> int:
        return len(self.capa_list) + len(self.induc_list)

    def _equation_size(self, transient: bool) -> int:
        return (
            len(self.node_list)
            + len(self.voltage_list)
            + len(self.capa_list)
            + len(self.induc_list)
            + (len(self.capa_list) + len(self.induc_list) if transient else 0)
        )

    def curr_index(self, partname: str, port: str) -> int:
        return self.curr_port_list.index((partname, port))

    def generate_code(
        self, transient: bool
    ) -> Callable[[dict[str, float]], ComputerBase]:
        n = self._equation_size(transient)
        cg = CodeGenerator(n, len(self.curr_port_list))

        counter = 0
        for part in self.parts:
            comp = part.component
            cname = part.name.replace("/", "_") + str(counter)
            nodes = part.connections
            valueCode = lambda x: cg.get_value_code(x)
            if isinstance(comp, Node2):
                kp = self.node_index(nodes[0])
                kn = self.node_index(nodes[1])
                curr_index_p = self.curr_index(part.name, "p")
                curr_index_n = self.curr_index(part.name, "n")

            if isinstance(comp, Voltage):
                codev = comp.codev(valueCode, cname)
                cg.add_to_cinit(codev.init)
                # the variable for the current through the voltage source
                k = self.voltage_index(part)
                cg.add_to_y_code(codev.pre)
                cg.add_ysum(kp, f"sol[{k}]")
                cg.add_ysum(kn, f"(-sol[{k}])")
                # equation k is for the voltage difference
                cg.add_ysum(k, f"(sol[{kp}] - sol[{kn}]) - ({codev.expr})")
                cg.add_dysum(kp, k, "1")
                cg.add_dysum(kn, k, "-1")
                cg.add_dysum(k, kp, "1")
                cg.add_dysum(k, kn, "-1")

                cg.add_to_cur_code(
                    [
                        f"res[{curr_index_p}] = sol[{k}]",
                        f"res[{curr_index_n}] = -(sol[{k}])",
                    ]
                )

            elif isinstance(comp, VoltageControlledVoltageSource):
                k = self.voltage_index(part)
                kin_p = self.node_index(nodes[0])
                kin_n = self.node_index(nodes[1])
                kout_p = self.node_index(nodes[2])
                kout_n = self.node_index(nodes[3])

                codev = comp.vcvcode(valueCode, cname, f"sol[{kin_p}] - sol[{kin_n}]")

                cg.add_to_cinit(codev.init)

                cg.add_ysum(kout_p, f"sol[{k}]")
                cg.add_ysum(kout_n, f"(-sol[{k}])")
                cg.add_dysum(kout_p, k, "1")
                cg.add_dysum(kout_n, k, "-1")

                cg.add_ysum(k, f"(sol[{kout_p}] - sol[{kout_n}]) - ({codev.expr})")

                cg.add_dysum(k, kout_p, "1")
                cg.add_dysum(k, kout_n, "-1")
                cg.add_dysum(k, kin_p, f"-{codev.dexpr}")
                cg.add_dysum(k, kin_n, f"{codev.dexpr}")

            elif isinstance(comp, Node2Current):
                # a simple componenet where the current depends onyl on the voltage
                # across the component
                code2 = comp.code2(valueCode, cname, f"sol[{kp}]- sol[{kn}]")
                cg.add_to_cinit(code2.component_init)
                cg.add_to_y_code(code2.current_init)
                cg.add_to_dy_code(code2.dcurrent_init)
                cg.add_ysum(kp, f"(-{code2.current})")
                cg.add_ysum(kn, f"({code2.current})")

                cg.add_dysum(kp, kp, f"(-{code2.dcurrent})")
                cg.add_dysum(kp, kn, f"{code2.dcurrent}")
                cg.add_dysum(kn, kp, f"{code2.dcurrent}")
                cg.add_dysum(kn, kn, f"(-{code2.dcurrent})")

                cg.add_to_cur_code(code2.current_init)
                cg.add_to_cur_code(
                    [
                        f"res[{curr_index_p}] = {code2.current}",
                        f"res[{curr_index_n}] =  -({code2.current})",
                    ]
                )
            elif isinstance(comp, VoltageControlledCurrentSource):
                kin_p = self.node_index(nodes[0])
                kin_n = self.node_index(nodes[1])
                kout_p = self.node_index(nodes[2])
                kout_n = self.node_index(nodes[3])

                codec = comp.vcccode(valueCode, cname, f"sol[{kin_p}] - sol[{kin_n}]")
                cg.add_to_cinit(codec.init)
                cg.add_ysum(kout_p, codec.expr)
                cg.add_ysum(kout_n, "-{codec.expr}")

                cg.add_dysum(kout_p, kin_p, codec.dexpr)
                cg.add_dysum(kout_p, kin_n, f"-{codec.dexpr}")

                cg.add_dysum(kout_n, kin_p, f"-{codec.dexpr}")
                cg.add_dysum(kout_n, kin_n, codec.dexpr)

            elif isinstance(comp, NPort):
                ports = comp.get_ports()
                voltages = {}
                i = 0
                for port in ports:
                    voltages[port] = f"sol[{self.port_node_index(part, port)}]"
                code = comp.code(cname, voltages)
                cg.add_to_cinit(code.component_init)
                cg.add_to_y_code(code.current_init)
                cg.add_to_dy_code(code.dcurrent_init)

                for port in ports:
                    pindex = self.port_node_index(part, port)
                    cg.add_ysum(pindex, code.current[port])
                    for portd in ports:
                        x = code.dcurrent[port][portd]
                        if not (x is None):
                            dpindex = self.port_node_index(part, portd)
                            cg.add_dysum(pindex, dpindex, x)

                cg.add_to_cur_code(code.current_init)
                for port in ports:
                    pindex = self.curr_index(part.name, port)
                    cu = code.current[port]
                    cg.add_to_cur_code([f"res[{pindex}]= -({cu})"])

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
                    # cg.add_ysum(k, f"sol[{kp}] - sol[{kn}] - state_vec[{sn}]")

                    cg.add_ysum(
                        sny, f"sol[{sny}] + h * sol[{k}]/{capa} - state_vec[{sn}]"
                    )

                    cg.add_dysum(kp, k, "1")
                    cg.add_dysum(kn, k, "-1")
                    cg.add_dysum(k, kp, "1")
                    cg.add_dysum(k, kn, "(-1)")
                    cg.add_dysum(k, sny, "(-1)")
                    cg.add_dysum(sny, sny, "1")
                    cg.add_dysum(sny, k, f"+h/{capa}")

                    cg.add_to_cur_code(
                        [
                            f"res[{curr_index_p}] = -(sol[{k}])",
                            f"res[{curr_index_n}] = sol[{k}]",
                        ]
                    )
                else:
                    # sol[k] current through capacitor, for working point condition is
                    # this current is 0
                    # this is a dummy equation
                    cg.add_ysum(k, f"sol[{k}]")
                    cg.add_dysum(k, k, "1")
                    cg.add_to_cur_code(
                        [
                            f"res[{curr_index_p}] = sol[{k}]",
                            f"res[{curr_index_n}] = -(sol[{k}])",
                        ]
                    )

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
                    cg.add_ysum(
                        sny,
                        f"sol[{sny}] - h *  (sol[{kp}] "
                        + f" - sol[{kn}])/{induc} - state_vec[{sn}]",
                    )

                    cg.add_dysum(kp, k, "(-1)")
                    cg.add_dysum(kn, k, "1")
                    cg.add_dysum(k, k, "1")
                    cg.add_dysum(k, sny, "-1")
                    cg.add_dysum(sny, sny, "1")
                    cg.add_dysum(sny, kp, f"-h /{induc}")
                    cg.add_dysum(sny, kn, f"h /{induc}")

                    cg.add_to_cur_code(
                        [
                            f"res[{curr_index_p}] = state_vec[{sn}]",
                            f"res[{curr_index_n}] = -state_vec[{sn}]",
                        ]
                    )
                else:
                    # sol[k] is the current through the inductor
                    # both nodes have the same voltage level
                    cg.add_ysum(k, f"sol[{kp}]- sol[{kn}]")
                    cg.add_ysum(kp, f"-sol[{k}]")  # current leaves node
                    cg.add_ysum(kn, f"(sol[{k}])")  # current enters node

                    cg.add_dysum(k, kp, "1")
                    cg.add_dysum(k, kn, "(-1)")
                    cg.add_dysum(kp, k, "(-1)")
                    cg.add_dysum(kn, k, "1")

                    cg.add_to_cur_code(
                        [
                            f"res[{curr_index_p}] = sol[{k}]",
                            f"res[{curr_index_n}] = -(sol[{k}])",
                        ]
                    )
            else:
                raise Exception("unknown component")

        for i in range(n):
            if i == 0:  # skip current equation for ground, force voltage to be 0
                cg.add_to_y_code(["res[0]=sol[0]"])
                cg.add_to_dy_code(["res[0][0]=1"])
            else:
                cg.add_to_y_code([f"res[{i}]=" + " + ".join(cg.ysum[i])])
                for j in range(n):
                    if len(cg.dysum[i][j]) > 0:
                        cg.add_to_dy_code(
                            [f"res[{i}][{j}]=" + " + ".join(cg.dysum[i][j])]
                        )
        cg.add_to_y_code(["return res", ""])
        cg.add_to_dy_code(["return res", ""])
        cg.add_to_cur_code(["return res"])

        cg.add_to_init([f"self.variables = [0] * {len(cg.variables)}"])
        vard = {}
        for var, k in cg.variables.items():
            vard[var.name] = k
            cg.add_to_init([f"self.variables[{k}] = {var.default}"])
        cg.add_to_init(
            [
                f"self.variable_map= {repr(vard)}",
                "for (k,v) in variables.items():",
                "     self.set_variable(k,v)",
            ]
        )

        x = [
            "    def set_variable(self, name, value):",
            "        if not name in self.variable_map:",
            "            raise Exception('unknown variable: ' +repr(name))",
            "        self.variables[self.variable_map[name]] = value",
            "",
        ]

        l = (
            cg.init
            + cg.component_init
            + [""]
            + x
            + cg.y_code
            + cg.dy_code
            + cg.cur_code
        )
        final_code = "\n".join(l)
        d: dict[str, Any] = {"ComputerBase": ComputerBase}

        exec(final_code, d)
        bla = d["bla"]
        return bla  # type: ignore

    def _compute_state_vec(
        self, capa_voltages: dict[str, float], induc_currents: dict[str, float]
    ) -> np_float_vec:
        state_vec = np.zeros(self.state_size())

        for part in self.parts:
            if isinstance(part.component, Capacitor):
                k = self.state_index(part)
                if part.name in capa_voltages:
                    state_vec[k] = capa_voltages[part.name]
                else:
                    raise Exception(f"no voltage given for capacitor {part.name}")
            if isinstance(part.component, Inductor):
                k = self.state_index(part)
                if part.name in induc_currents:
                    state_vec[k] = induc_currents[part.name]
                else:
                    raise Exception(f"no  current given for inductor {part.name}")

        return state_vec

    def _compute_currents(
        self, bla: ComputerBase, time: float, sol: np_float_vec, state_vec: np_float_vec
    ) -> dict[str, float]:
        res: dict[str, float] = {}
        cv = bla.currents(time, sol, state_vec)
        for name, port in self.curr_port_list:
            res[name + "." + port] = cv[self.curr_index(name, port)]
        return res

    def _compute_voltages(self, solution_vec: np_float_vec) -> dict[str, float]:
        voltages: dict[str, float] = {}
        for part in self.parts:
            ports = part.component.get_ports()
            nodes = part.connections
            assert len(ports) == len(nodes)
            for port, node in zip(ports, nodes):
                k = self.node_index(node)
                port_name = part.name + "." + port
                voltages[port_name] = solution_vec[k]
        for node in self.node_list:
            k = self.node_index(node)
            voltages[node] = solution_vec[k]
        return voltages

    def analyze(
        self,
        maxit: int = 20,
        start_solution_vec: Optional[np_float_vec] = None,
        abstol: float = 1e-8,
        reltol: float = 1e-6,
        variables: Optional[dict[str, float]] = None,
        capa_voltages: Optional[dict[str, float]] = None,
        induc_currents: Optional[dict[str, float]] = None,
        start_voltages: Optional[dict[str, float]] = None,
        time: float = 0,
        transient: bool = False,
        compute_cond: bool = False,
        verbose: bool = False,
    ) -> Union[str, Result]:
        n = self._equation_size(transient)
        if start_solution_vec is None:
            if start_voltages is None:
                if verbose:
                    print("starting from zero")
                solution_vec0 = np.zeros(n)
            else:
                if verbose:
                    print("starting from start voltages")
                solution_vec0 = np.zeros(n)
                for vk in start_voltages:
                    n = self.port_node_indexes[vk]
                    solution_vec0[n] = start_voltages[vk]
        else:
            solution_vec0 = start_solution_vec

        solution_vec = solution_vec0

        if transient:
            state_vec = self._compute_state_vec(
                capa_voltages or {}, induc_currents or {}
            )
        else:
            state_vec = np.zeros(1)  # None

        bla = self.generate_code(transient)
        computer = bla(variables or {})

        def f(x: np_float_vec) -> np_float_vec:
            assert not isinstance(x, str)
            return computer.y(time, x, state_vec, 0)

        def Df(x: np_float_vec) -> np_float_vec:
            return computer.dy(time, x, state_vec, 0)

        res = solving.solve_alfa(
            solution_vec, f, Df, abstol, reltol, maxit, verbose=verbose
        )
        if not isinstance(res, str):
            (sol, y, dfx, iterations, norm_y) = res
            if compute_cond:
                cond = np.linalg.cond(dfx, "fro")
            else:
                cond = None
            norm_y = float(np.linalg.norm(y))
            currents = self._compute_currents(computer, time, sol, state_vec)
            return Result(self.parts, self, iterations, sol, y, norm_y, cond, currents)
        else:
            return res

    def solve_internal(
        self,
        time: float,
        maxit: int,
        start_sol: np_float_vec,
        state_vec: np_float_vec,
        abstol: float,
        reltol: float,
        c: ComputerBase,
        compute_cond: bool,
        h: float,
    ) -> Union[
        str,
        tuple[
            tuple[np_float_vec, np_float_vec, float, Optional[float], int],
            dict[str, float],
            dict[str, float],
        ],
    ]:
        def f(x: np_float_vec) -> np_float_vec:
            return c.y(time, x, state_vec, h)

        def Df(x: np_float_vec) -> np_float_vec:
            return c.dy(time, x, state_vec, h)

        res = solving.solve(start_sol, f, Df, abstol, reltol, maxit)
        if isinstance(res, str):
            return res
        if compute_cond:
            cond = float(np.linalg.cond(res.dfx, "fro"))
        else:
            cond = None
        currents = self._compute_currents(c, time, res.x, state_vec)
        voltages = self._compute_voltages(res.x)
        return ((res.x, res.y, res.norm_y, cond, res.iterations), voltages, currents)

    def transient(
        self,
        maxtime: float,
        timestep: float,
        maxit: int = 20,
        start_solution_vec: Optional[np_float_vec] = None,
        abstol: float = 1e-8,
        reltol: float = 1e-6,
        variables: Optional[dict[str, float]] = None,
        capa_voltages: Optional[dict[str, float]] = None,
        induc_currents: Optional[dict[str, float]] = None,
        start_voltages: Optional[dict[str, float]] = None,
        compute_cond: bool = False,
    ) -> TransientResult:
        capa_voltages = capa_voltages or {}
        induc_currents = induc_currents or {}

        time = 0.0
        max_timestep = timestep

        min_timestep = timestep / 10000.0
        res0 = self.analyze(
            maxit=maxit,
            start_solution_vec=start_solution_vec,
            capa_voltages=capa_voltages,
            induc_currents=induc_currents,
            variables=variables,
            start_voltages=start_voltages,
            time=time,
            abstol=abstol,
            reltol=reltol,
            transient=True,
            compute_cond=compute_cond,
        )
        if isinstance(res0, str):
            raise Exception("can not find inital solution")

        state_vec = self._compute_state_vec(capa_voltages, induc_currents)

        voltage_list: list[dict[str, float]] = []
        current_list: list[dict[str, float]] = []
        time_list: list[float] = []

        sol = res0.solution_vec
        bla = self.generate_code(True)
        computer = bla(variables or {})
        while time < maxtime:
            res = self.solve_internal(
                time,
                maxit,
                sol,
                state_vec,
                abstol,
                reltol,
                computer,
                compute_cond,
                timestep,
            )
            if isinstance(res, str):
                print("res: " + res)
                timestep = timestep / 2
                if timestep < min_timestep * 2:
                    n = self._equation_size(True)
                    #                    sol = np.zeros(n) + np.random.rand(n)
                    sol = np.random.rand(n) * 10
                    timestep = timestep * 4
                if timestep < min_timestep:
                    print(f"fail at time {time}: {res}, stepisze={timestep}")
                    break
                print("dec step", time, timestep)
                continue
            a = timestep * 1.05
            if a < max_timestep:
                timestep = a
                print("inc step", time, timestep)
            (
                (sol_x, _sol_y, _sol_ny, _sol_cond, _sol_iterations),
                voltages,
                currents,
            ) = res
            time_list.append(time)
            voltage_list.append(voltages)
            current_list.append(currents)

            sol = sol_x

            for part in self.parts:
                comp = part.component
                if isinstance(comp, (Capacitor, Inductor)):
                    k = self.state_index(part)
                    state_vec[k] = sol[self.state_index_y(part)]

            time += timestep
        return TransientResult(time_list, voltage_list, current_list)

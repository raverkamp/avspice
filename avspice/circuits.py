"""circuit definition and components"""

import collections
from typing import Optional, Union, Callable, TypeAlias

import numbers
import pprint as pp
from . import util

from .variable import Variable

ValueCode: TypeAlias = Callable[[Union[Variable, float]], str]

# for codee generation hsi returne by Node2 components like resistors, diodes and current sources
Node2Code = collections.namedtuple(
    "Node2Code",
    [
        # initialization code for component
        "component_init",
        # initialization code for  current computation
        "current_init",
        # expression of current
        "current",
        # initialization code for derivative of current computation
        "dcurrent_init",
        # expression for diff of current
        "dcurrent",
    ],
)

NodeNCode = collections.namedtuple(
    "NodeNCode",
    [
        # initialization code for component
        "component_init",
        # initialization code for  current computation
        "current_init",
        # expression of current
        "current",
        # initialization code for derivative of current computation
        "dcurrent_init",
        # expression for diff of current
        "dcurrent",
    ],
)

VoltageCode = collections.namedtuple("VoltageCode", ["init", "pre", "expr"])

VCVCode = collections.namedtuple("VCVCode", ["init", "expr", "dexpr"])

VCCCode = collections.namedtuple("VCCCode", ["init", "expr", "dexpr"])


class Component:
    """Component in a electrical network, e.g. resistor, current source, node"""

    name: str

    def __init__(self, name: str):
        assert isinstance(name, str)
        self.name = name

    def get_ports(self) -> list[str]:
        """return the ports of this component"""
        raise NotImplementedError("method 'get_ports' is not implemented")


class NPort(Component):
    """Nport"""

    def get_ports(self) -> list[str]:
        """return the ports of this component"""
        raise NotImplementedError("method 'get_ports' is not implemented")

    def code(self, name: str, voltages: dict[str, str]) -> NodeNCode:
        raise NotImplementedError("method 'code' is not implemented")


class Node2(Component):
    """a component with just two ports"""

    #  __init__ is the same as for Component

    def get_ports(self) -> list[str]:
        return ["p", "n"]


class Node2Current(Node2):
    """subclass for Node2 current

    components which conduct a current based solely on the applied voltage
    """

    def code2(self, valueCode: ValueCode, cname: str, dvname: str) -> Node2Code:
        raise NotImplementedError("method 'code' is not implemented")


class Resistor(Node2Current):
    """resistor"""

    _ohm: Union[float, Variable]

    def __init__(self, name: str, ohm: Union[float, Variable]):
        super().__init__(name)
        self._ohm = ohm

    def get_resistance(self) -> Union[Variable, float]:
        return self._ohm

    def __repr__(self) -> str:
        return f"<Resistor {self._ohm}>"

    def code2(self, valueCode: ValueCode, cname: str, dvname: str) -> Node2Code:
        r = valueCode(self.get_resistance())
        G = f"self.{cname}_G"
        return Node2Code(
            component_init=[f"{G}=1/{r}"],
            current_init=[],
            current=f"{G}*({dvname})",
            dcurrent_init=[],
            dcurrent=f"{G}",
        )


class Current(Node2Current):
    """current source"""

    def __init__(self, name: str, amp: Union[Variable, float]):
        super().__init__(name)
        self.amp = amp

    def __repr__(self) -> str:
        return f"<Current {self.name}>"

    def code2(self, valueCode: ValueCode, cname: str, dvname: str) -> Node2Code:
        _ = cname
        _ = dvname
        x = valueCode(self.amp)
        return Node2Code(
            component_init=[],
            current_init=[],
            current=f"-{x}",
            dcurrent_init=[],
            dcurrent="0",
        )


class Voltage(Node2):
    """voltage source"""

    def __init__(self, name: str, volts: Union[Variable, float]):
        super().__init__(name)
        assert isinstance(
            volts, (numbers.Number, Variable)
        ), "volts must be a variable or a number"
        self._volts = volts

    def __repr__(self) -> str:
        return f"<Voltage {self._volts}>"

    def codev(self, valueCode: ValueCode, cname: str) -> VoltageCode:
        v = valueCode(self._volts)
        init = [f"self.{cname} = NVoltage({v})"]
        return VoltageCode(
            init, [f"{cname}_voltage = self.{cname}.voltage(time)"], f"{cname}_voltage"
        )


class SineVoltage(Voltage):
    """sine voltage"""

    def __init__(self, name: str, volts: float, frequency: float):
        super().__init__(name, volts)
        self._frequency = frequency

    def codev(self, valueCode: ValueCode, cname: str) -> VoltageCode:
        v = valueCode(self._volts)
        f = valueCode(self._frequency)
        init = [f"self.{cname} = NSineVoltage({v}, {f})"]
        return VoltageCode(
            init, [f"{cname}_voltage = self.{cname}.voltage(time)"], f"{cname}_voltage"
        )


class SawVoltage(Voltage):
    """saw voltage"""

    def __init__(self, name: str, volts: float, frequency: float):
        super().__init__(name, volts)
        self._frequency = frequency

    def codev(self, valueCode: ValueCode, cname: str) -> VoltageCode:
        v = valueCode(self._volts)
        init = [f"self.{cname} = NSawVoltage({v}, {self._frequency})"]
        return VoltageCode(
            init, [f"{cname}_voltage = self.{cname}.voltage(time)"], f"{cname}_voltage"
        )


class PwmVoltage(Voltage):
    """pwm voltage"""

    def __init__(self, name: str, volts: float, frequency: float, duty: float = 0.5):
        super().__init__(name, volts)
        self._frequency = frequency
        self._duty = duty

    def codev(self, valueCode: ValueCode, cname: str) -> VoltageCode:
        v = valueCode(self._volts)
        init = [f"self.{cname} = NPwmVoltage({v}, {self._frequency}, {self._duty})"]
        return VoltageCode(
            init, [f"{cname}_voltage = self.{cname}.voltage(time)"], f"{cname}_voltage"
        )


class PieceWiseLinearVoltage(Voltage):
    """piecewise linear voltage"""

    def __init__(self, name: str, pairs: list[tuple[float, Union[Variable, float]]]):
        super().__init__(name, 0)
        self.pairs = list(pairs)

    def codev(self, valueCode: ValueCode, cname: str) -> VoltageCode:
        a = list(self.pairs)
        a.sort(key=lambda x: x[0])
        vx = list(x for (x, y) in a)

        vy = "[" + ", ".join(list(valueCode(y) for (x, y) in a)) + "]"

        init = [f"self.{cname} = NPieceWiseLinearVoltage({vx},  {vy})"]
        return VoltageCode(
            init, [f"{cname}_voltage = self.{cname}.voltage(time)"], f"{cname}_voltage"
        )


class PeriodicPieceWiseLinearVoltage(Voltage):
    """periodic piecewise linear voltage"""

    def __init__(
        self,
        name: str,
        period: float,
        pairs: list[tuple[float, Union[Variable, float]]],
        freq_mul: Union[Variable, float],
        volt_mul: Union[Variable, float],
        shift: Union[Variable, float] = 0,
    ):
        super().__init__(name, 0)
        self.pairs = list(pairs)
        assert isinstance(period, (float, int))
        self.period = period
        assert isinstance(freq_mul, (Variable, int, float))
        self.freq_mul = freq_mul
        assert isinstance(volt_mul, (Variable, int, float))
        self.volt_mul = volt_mul
        assert isinstance(shift, (Variable, int, float))
        self.shift = shift

    def codev(self, valueCode: ValueCode, cname: str) -> VoltageCode:
        a = list(self.pairs)
        a.sort(key=lambda x: x[0])
        vx = list(x for (x, y) in a)

        vy = "[" + ", ".join(list(valueCode(y) for (x, y) in a)) + "]"

        init = [
            f"self.{cname} = NPeriodicPieceWiseLinearVoltage({self.period}, {vx},"
            + f"  {vy}, {valueCode(self.freq_mul)}, {valueCode(self.volt_mul)}, {valueCode(self.shift)})"
        ]
        return VoltageCode(
            init, [f"{cname}_voltage = self.{cname}.voltage(time)"], f"{cname}_voltage"
        )


class VoltageControlledVoltageSource(Component):
    def __init__(self, name: str):
        super().__init__(name)

    def get_ports(self) -> list[str]:
        return ["vinp", "vinn", "voutp", "voutn"]

    def ncompo_code(self) -> str:
        return f"NLinearVoltageControlledVoltageSource(1.0)"

    def vcvcode(self, valueCode: ValueCode, cname: str, dvname: str) -> VCVCode:
        component_init = [f"self.{cname} = " + self.ncompo_code()]
        v_code = f"self.{cname}.voltage({dvname})"
        dv_code = f"self.{cname}.dvoltage({dvname})"
        return VCVCode(component_init, v_code, dv_code)

    def __repr__(self) -> str:
        return f"<VoltageControlledVoltageSource {self.name}>"


class LinearVoltageControlledVoltageSource(VoltageControlledVoltageSource):
    def __init__(self, name: str, gain: float):
        super().__init__(name)
        self.gain = gain

    def ncompo_code(self) -> str:
        return f"NLinearVoltageControlledVoltageSource({self.gain})"


class VoltageControlledCurrentSource(Component):
    def __init__(self, name: str):
        super().__init__(name)

    def get_ports(self) -> list[str]:
        return ["vinp", "vinn", "ioutp", "ioutn"]

    def ncompo_code(self) -> str:
        return f"NLinearVoltageControlledCurrentSource(1.0)"

    def vcccode(self, valueCode: ValueCode, cname: str, dvname: str) -> VCVCode:
        component_init = [f"self.{cname} = " + self.ncompo_code()]
        i_code = f"self.{cname}.current({dvname})"
        di_code = f"self.{cname}.dcurrent({dvname})"
        return VCCCode(component_init, i_code, di_code)

    def __repr__(self) -> str:
        return f"<VoltageControlledCurrentSource {self.name}>"


class Diode(Node2Current):
    """solid state diode"""

    def __init__(
        self,
        name: str,
        Is: float,
        Nut: float,
        lcut_off: float = -40,
        rcut_off: float = 40,
    ):
        super().__init__(name)
        self.Is = Is
        self.Nut = Nut

        self.lcut_off = lcut_off
        self.rcut_off = rcut_off

    def code2(self, valueCode: ValueCode, cname: str, dvname: str) -> Node2Code:
        _ = valueCode
        return Node2Code(
            component_init=[
                f"self.{cname} = "
                + f" NDiode({self.Is},{self.Nut},{self.lcut_off},{self.rcut_off})"
            ],
            current_init=[],
            current=f"self.{cname}.current({dvname})",
            dcurrent_init=[],
            dcurrent=f"self.{cname}.diff_current({dvname})",
        )

    def __repr__(self) -> str:
        return f"<Diode {self.name}>"


class ZDiode(Node2Current):
    """solid state diode"""

    def __init__(
        self,
        name: str,
        vcut: float,
        Is: float,
        Nut: float,
        IsZ: Optional[float] = None,
        NutZ: Optional[float] = None,
        lcut_off: float = -40,
        rcut_off: float = 40,
    ):
        super().__init__(name)
        assert isinstance(name, str)
        assert isinstance(vcut, numbers.Number)

        self.Is = Is
        self.Nut = Nut
        self.vcut = vcut
        self.IsZ = Is if IsZ is None else IsZ
        self.NutZ = Nut if NutZ is None else NutZ

        self.lcut_off = lcut_off
        self.rcut_off = rcut_off

    def code2(self, valueCode: ValueCode, cname: str, dvname: str) -> Node2Code:
        _ = valueCode
        return Node2Code(
            component_init=[
                f"self.{cname} = NZDiode({self.vcut}, {self.Is},{self.Nut},"
                + f"{self.IsZ}, {self.NutZ}, {self.lcut_off},{self.rcut_off})"
            ],
            current_init=[],
            current=f"self.{cname}.current({dvname})",
            dcurrent_init=[],
            dcurrent=f"self.{cname}.diff_current({dvname})",
        )

    def __repr__(self) -> str:
        return f"<Diode {self.name}>"


class Capacitor(Node2):
    """a capacitor"""

    def __init__(self, name: str, capa: float):
        super().__init__(name)
        self._capa = capa

    def get_capacitance(self) -> float:
        return self._capa

    def __repr__(self) -> str:
        return f"<Capacitor {self.name}>"


class Inductor(Node2):
    """inductor"""

    def __init__(self, name: str, induc: float):
        super().__init__(name)
        self.induc = induc

    def get_inductance(self) -> float:
        return self.induc

    def __repr__(self) -> str:
        return f"<Inductor {self.name}>"


class PNPTransistor(NPort):

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

    def __init__(
        self,
        name: str,
        IS: float,
        VT: float,
        beta_F: float,
        beta_R: float,
        cutoff: float = 40,
    ):
        super().__init__(name)
        self.IS = IS
        self.VT = VT
        self.beta_F = beta_F
        self.beta_R = beta_R

        self.lcutoff = -cutoff
        self.rcutoff = cutoff

    def __repr__(self) -> str:
        return f"<NPNTransistor {self.name}>"

    def get_ports(self) -> list[str]:
        return ["B", "C", "E"]

    def code(self, name: str, voltages: dict[str, str]) -> NodeNCode:
        prefix = name
        me = "self." + prefix + "_"
        initt = [
            f"{me} = NPNPTransistor({self.IS}, {self.VT}, {self.beta_F},"
            + f" {self.beta_R}, {self.lcutoff}, {self.rcutoff})"
        ]
        vbe = f"{prefix}_vbe"
        vbc = f"{prefix}_vbc"

        nb = voltages["B"]
        ne = voltages["E"]
        nc = voltages["C"]

        cinit = [f"{vbe} = {nb}- {ne}", f"{vbc} = {nb}- {nc}"]

        currents = {
            "B": f"-{me}.IB({vbe}, {vbc})",
            "E": f"-{me}.IE({vbe}, {vbc})",
            "C": f"-{me}.IC({vbe}, {vbc})",
        }

        dinit = cinit

        dcurrent = {
            "B": {
                "B": f"-{me}.d_IB_vbe({vbe})- {me}.d_IB_vbc({vbc})",
                "E": f"{me}.d_IB_vbe({vbe})",
                "C": f"{me}.d_IB_vbc({vbc})",
            },
            "E": {
                "B": f"-{me}.d_IE_vbe({vbe}) - {me}.d_IE_vbc({vbc})",
                "E": f"{me}.d_IE_vbe({vbe})",
                "C": f"{me}.d_IE_vbc({vbc})",
            },
            "C": {
                "B": f"-{me}.d_IC_vbe({vbe}) - {me}.d_IC_vbc({vbc})",
                "E": f"{me}.d_IC_vbe({vbe})",
                "C": f"{me}.d_IC_vbc({vbc})",
            },
        }

        return NodeNCode(
            component_init=initt,
            current_init=cinit,
            current=currents,
            dcurrent_init=dinit,
            dcurrent=dcurrent,
        )


class FET(NPort):
    """a FET"""

    def __init__(self, name: str, vth: float):
        super().__init__(name)
        self.vth = vth

    def get_ports(self) -> list[str]:
        return ["G", "D", "S"]

    def code(self, name: str, voltages: dict[str, str]) -> NodeNCode:
        prefix = name
        me = "self." + prefix + "_"
        initt = [f"{me} = NFET({self.vth})"]

        vgs = f"{prefix}_vgs"
        vds = f"{prefix}_vds"

        ng = voltages["G"]
        nd = voltages["D"]
        ns = voltages["S"]

        cinit = [f"{vgs}  = {ng} - {ns}", f"{vds} = {nd} - {ns}"]

        currents = {
            "G": "0",
            "S": f"{me}.IS({vgs}, {vds})",
            "D": f"(-{me}.IS({vgs}, {vds}))",
        }

        dinit = cinit
        #  g d s
        dcurrent = {
            "G": {"G": "0", "S": "0", "D": "0"},
            "S": {
                "G": f"{me}.d_IS_vgs({vgs},{vds})",
                "D": f"{me}.d_IS_vds({vgs},{vds})",
                "S": f"(-{me}.d_IS_vgs({vgs},{vds}) - {me}.d_IS_vds({vgs},{vds}))",
            },
            "D": {
                "G": f"(-{me}.d_IS_vgs({vgs},{vds}))",
                "D": f"(-{me}.d_IS_vds({vgs},{vds}))",
                "S": f"({me}.d_IS_vgs({vgs},{vds}) + {me}.d_IS_vds({vgs},{vds}))",
            },
        }

        return NodeNCode(
            component_init=initt,
            current_init=cinit,
            current=currents,
            dcurrent_init=dinit,
            dcurrent=dcurrent,
        )


class JFET(NPort):
    """a JFET"""

    def __init__(self, name: str, vth: float, beta: float, lambda_: float):
        super().__init__(name)
        self.vth = vth
        self.vth = vth
        self.beta = beta
        self.lambda_ = lambda_

    def get_ports(self) -> list[str]:
        return ["G", "D", "S"]

    def code(self, name: str, voltages: dict[str, str]) -> NodeNCode:
        prefix = name
        me = "self." + prefix + "_"
        initt = [f"{me} = NJFETn({self.vth},{self.beta}, {self.lambda_})"]

        vgs = f"{prefix}_vgs"
        vds = f"{prefix}_vds"

        ng = voltages["G"]
        nd = voltages["D"]
        ns = voltages["S"]

        cinit = [f"{vgs}  = {ng} - {ns}", f"{vds} = {nd} - {ns}"]

        curr = {
            "G": "0",
            "S": f"{me}.IS({vgs}, {vds})",
            "D": f"(-{me}.IS({vgs}, {vds}))",
        }

        dinit = cinit
        #  g d s
        dcurr = {
            "G": {"G": 0, "S": "0", "D": "0"},
            "S": {
                "G": f"{me}.d_IS_vgs({vgs},{vds})",
                "D": f"{me}.d_IS_vds({vgs},{vds})",
                "S": f"(-{me}.d_IS_vgs({vgs},{vds}) - {me}.d_IS_vds({vgs},{vds}))",
            },
            "D": {
                "G": f"(-{me}.d_IS_vgs({vgs},{vds}))",
                "D": f"(-{me}.d_IS_vds({vgs},{vds}))",
                "S": f"({me}.d_IS_vgs({vgs},{vds}) + {me}.d_IS_vds({vgs},{vds}))",
            },
        }

        return NodeNCode(
            component_init=initt,
            current_init=cinit,
            current=curr,
            dcurrent_init=dinit,
            dcurrent=dcurr,
        )


class NPNTransistor(NPort):
    """an NPN Transistir implemented as NPort

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

    def __init__(
        self,
        name: str,
        IS: float,
        VT: float,
        beta_F: float,
        beta_R: float,
        cutoff: float = 40,
    ):
        super().__init__(name)
        self.IS = IS
        self.VT = VT
        self.beta_F = beta_F
        self.beta_R = beta_R

        self.lcutoff = -cutoff
        self.rcutoff = cutoff

    def __repr__(self) -> str:
        return f"<NPNTransistor {self.name}>"

    def get_ports(self) -> list[str]:
        return ["B", "C", "E"]

    def code(self, name: str, voltages: dict[str, str]) -> NodeNCode:
        prefix = name
        me = "self." + prefix + "_"
        initt = [
            f"{me} = NNPNTransistor({self.IS}, {self.VT},"
            + f" {self.beta_F}, {self.beta_R}, {self.lcutoff}, {self.rcutoff})"
        ]
        vbe = f"{prefix}_vbe"
        vbc = f"{prefix}_vbc"

        nb = voltages["B"]
        ne = voltages["E"]
        nc = voltages["C"]

        cinit = [f"{vbe} = {nb}- {ne}", f"{vbc} = {nb}- {nc}"]

        currents = {
            "B": f"(-{me}.IB({vbe}, {vbc}))",
            "E": f"{me}.IE({vbe}, {vbc})",
            "C": f"(-{me}.IC({vbe}, {vbc}))",
        }

        ib_vbe = f"{prefix}_ib_vbe"
        ib_vbc = f"{prefix}_ib_vbc"

        ie_vbe = f"{prefix}_ie_vbe"
        ie_vbc = f"{prefix}_ie_vbc"

        ic_vbe = f"{prefix}_ic_vbe"
        ic_vbc = f"{prefix}_ic_vbc"

        dinit = cinit + [
            f"{ib_vbe}= {me}.d_IB_vbe({vbe})",
            f"{ib_vbc}= {me}.d_IB_vbc({vbc})",
            f"{ie_vbe}= {me}.d_IE_vbe({vbe})",
            f"{ie_vbc}= {me}.d_IE_vbc({vbc})",
            f"{ic_vbe}= {me}.d_IC_vbe({vbe})",
            f"{ic_vbc}= {me}.d_IC_vbc({vbc})",
        ]

        dcurrent = {
            "B": {"B": f"(-{ib_vbe}-{ib_vbc})", "E": ib_vbe, "C": ib_vbc},
            "E": {
                "B": f"{ie_vbe} + {ie_vbc}",
                "E": f"(-{ie_vbe})",
                "C": f"(-{ie_vbc})",
            },
            "C": {"B": f"(-{ic_vbe} - {ic_vbc})", "E": f"{ic_vbe}", "C": f"{ic_vbc}"},
        }

        return NodeNCode(
            component_init=initt,
            current_init=cinit,
            current=currents,
            dcurrent_init=dinit,
            dcurrent=dcurrent,
        )


def check_valid_name(name):
    if name == "" or not name.isascii() or not name.isprintable():
        raise ValueError("name must be not empty, ascii and printable")
    if "/" in name:
        raise ValueError("'/' is not allowed in names")


# utility class for for networks
# name: a name of the component in a network,
# component:  the component itself, e.g. transistor BC238, a resistor with a specific resistance
# connections: the mapping  of the ports of the component to nodes in the network
Part = collections.namedtuple("Part", ("name", "component", "connections"))


class Network:
    """this class describes the toplogy of an electrical network
    It only contains the topology"""

    def __init__(self) -> None:
        self.parts: list[Part] = []  # the list of parts
        self.node_list: list[str] = []  # the list of nodes
        self.part_dict: dict[str, Part] = {}  # mapping from part name to parts

    def add_component(self, name: str, comp: Component, nodes: list[str]) -> None:
        assert isinstance(name, str), "name parameter must be a string"
        check_valid_name(name)
        assert isinstance(comp, Component), "component parameter must be a component"
        if name in self.part_dict:
            raise ValueError(f"a part with name {name} already exists")
        ports = comp.get_ports()
        if not len(ports) == len(nodes):
            raise ValueError(
                f"for part {name} #ports={len(ports)} and #connections={len(nodes)} do not match"
            )

        # it is possible and OK that there are duplicates in nodes!
        for node in nodes:
            assert isinstance(node, str)
            check_valid_name(node)

        for node in nodes:
            if not node in self.node_list:
                self.node_list.append(node)
        part = Part(name, comp, nodes)
        self.parts.append(part)
        self.part_dict[name] = part

    def addR(self, name: str, ohm: float, p: str, n: str) -> None:
        """add a curent source"""
        c = Resistor(name, ohm)
        self.add_component(name, c, [p, n])

    def addC(self, name: str, amp: Union[Variable, float], p: str, n: str) -> None:
        """add a curent source"""
        c = Current(name, amp)
        self.add_component(name, c, [p, n])

    def addV(self, name: str, volts: Union[Variable, float], p: str, n: str) -> None:
        v = Voltage(name, volts)
        self.add_component(name, v, [p, n])

    def addSineV(
        self, name: str, volts: float, frequency: float, p: str, n: str
    ) -> None:
        v = SineVoltage(name, volts, frequency)
        self.add_component(name, v, [p, n])

    def addSawV(
        self, name: str, volts: float, frequency: float, p: str, n: str
    ) -> None:
        v = SawVoltage(name, volts, frequency)
        self.add_component(name, v, [p, n])

    def addD(self, name: str, Is: float, Nut: float, p: str, n: str) -> None:
        d = Diode(name, Is, Nut)
        self.add_component(name, d, [p, n])

    def addCapa(self, name: str, capa: float, p: str, n: str) -> None:
        c = Capacitor(name, capa)
        self.add_component(name, c, [p, n])

    def addInduc(self, name: str, induc: float, p: str, n: str) -> None:
        indu = Inductor(name, induc)
        self.add_component(name, indu, [p, n])

    def add_subcircuit(
        self, name: str, subcircuit: "SubCircuit", nodes: list[str]
    ) -> None:
        assert isinstance(subcircuit, SubCircuit)

        assert util.is_str_seq(nodes)
        c = SubCircuitComponent(subcircuit)
        self.add_component(name, c, nodes)

    def add(
        self, name: str, x: Union["SubCircuit", Component], nodes: list[str]
    ) -> None:
        assert isinstance(name, str)
        if isinstance(x, SubCircuit):
            self.add_subcircuit(name, x, nodes)
        elif isinstance(x, Component):
            self.add_component(name, x, nodes)
        else:
            raise ValueError("expecting a Component or Subcircuit as second argument")


class Circuit(Network):
    """toplevel circuit"""

    def __init__(self) -> None:
        super().__init__()
        self.node_list.append("0")


class SubCircuit(Network):
    """a sub circuit, reuse  in circuits"""

    def __init__(self, export_nodes: list[str]):
        super().__init__()
        assert util.is_str_seq(export_nodes), "nodes must be a sequence of strings"
        self.export_nodes = list(export_nodes)


class SubCircuitComponent(Component):
    """wrapper around a subcircuit in circuit"""

    def __init__(self, subcircuit: SubCircuit):
        assert isinstance(subcircuit, SubCircuit)
        super().__init__("nix")
        self.subcircuit = subcircuit

    def get_ports(self) -> list[str]:
        return self.subcircuit.export_nodes

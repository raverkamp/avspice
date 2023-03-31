"""class variable  and codegenerator"""
from collections.abc import Iterator
import numbers
from typing import Optional, Any, Callable, Union

class Variable:
    """a variable"""
    name:str
    def __init__(self, name:str, default:Optional[float]=None) -> None:
        self.name = name
        assert default is None or isinstance(default, numbers.Number)
        self.default = default

    def __repr__(self)->str:
        return f"<Variable {self.name}, {self.default}>"

class CodeGenerator:
    """utility class for code generation"""

    def __init__(self, n:int, n_curr_ports:int, transient:bool)->None:
        self.n = n
        self.n_curr_ports = n_curr_ports
        #        if transient:
        h_par = ", h"
        #       else:
        #          h_par = ""
        self.component_init:list[str] = []
        self.init = [
            "def bla(variables):",
            "   return Computer(variables)",
            "",
            "class Computer(ComputerBase):",
            "    def __init__(self, variables):",
            "        from avspice.ncomponents import NDiode, NZDiode, NNPNTransistor,"
                     + " NPNPTransistor, NFET, NJFETn,"
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
        self.ysum:list[list[str]]=[]
        for _ in range(self.n):
            self.ysum.append([])
        self.dysum:list[list[list[str]]]=[]
        for _ in range(self.n):
            x:list[list[str]] = []
            for _ in range(n):
                x.append([])
            self.dysum.append(x)
        self.variables:dict[Variable,int] = {}

    def get_var_code(self, variable:Variable)->str:
        assert isinstance(variable, Variable), "variable parameter must be instance of variable"
        if variable in self.variables:
            k = self.variables[variable]
        else:
            k = len(self.variables)
            self.variables[variable] = k
        return f"self.variables[{k}]"

    def get_value_code(self, x:Union[Variable,float])->str:
        if  isinstance(x, Variable):
            return self.get_var_code(x)
        elif isinstance(x, (float, int)):
            return str(x)
        else:
            raise Exception("wrong type for scalar value:" + str(x))

    def add_ysum(self, k:int, e:str)->None:
        self.ysum[k].append(e)

    def add_dysum(self, k:int, j:int, e:str)->None:
        self.dysum[k][j].append(e)

    def add_to_method(self, m:list[str], lines_or_str:Union[str,list[str]])->None:
        if isinstance(lines_or_str, str):
            l=[lines_or_str]
        else:
            l=lines_or_str
        for x in l:
            assert isinstance(x, str)
            m.append("        " + x)

    def add_to_cinit(self,l:list[str])->None:
        self.add_to_method(self.component_init, l)

    def add_to_init(self,l:list[str])->None:
        self.add_to_method(self.init, l)

    def add_to_y_code(self, l:list[str])->None:
        self.add_to_method(self.y_code, l)

    def add_to_dy_code(self, l:list[str])->None:
        self.add_to_method(self.dy_code, l)

    def add_to_cur_code(self, l:list[str])->None:
        self.add_to_method(self.cur_code, l)

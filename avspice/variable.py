"""class variable"""

import numbers
from typing import Optional


class Variable:
    """a variable"""

    name: str

    def __init__(self, name: str, default: Optional[float] = None) -> None:
        self.name = name
        assert default is None or isinstance(default, numbers.Number)
        self.default = default

    def __repr__(self) -> str:
        return f"<Variable {self.name}, {self.default}>"

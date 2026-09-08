"""Operational model for the sys module.

Data only: the interpreter-state attributes a verification harness reads. The
standard streams (stdout/stderr/stdin), sys.exit and the frame/trace
introspection API are not modelled.

Values describe the 64-bit little-endian target ESBMC's C models assume and
the IEEE-754 binary64 float the frontend maps `float` onto; they match CPython
on that target. `platform` and `version_info` cannot be derived that way and
name the target the models assume, not the host interpreter.
"""
# The attribute names are the stdlib's API, not names this module chooses.
# pylint: disable=invalid-name,too-few-public-methods,too-many-instance-attributes


class _FloatInfo:
    """sys.float_info — IEEE-754 binary64 limits."""

    def __init__(self) -> None:
        self.max: float = 1.7976931348623157e+308
        self.max_exp: int = 1024
        self.max_10_exp: int = 308
        self.min: float = 2.2250738585072014e-308
        self.min_exp: int = -1021
        self.min_10_exp: int = -307
        self.dig: int = 15
        self.mant_dig: int = 53
        self.epsilon: float = 2.220446049250313e-16
        self.radix: int = 2
        self.rounds: int = 1


class _VersionInfo:
    """sys.version_info — the Python level the frontend accepts."""

    def __init__(self) -> None:
        self.major: int = 3
        self.minor: int = 12
        self.micro: int = 0
        self.releaselevel: str = "final"
        self.serial: int = 0


float_info: _FloatInfo = _FloatInfo()
version_info: _VersionInfo = _VersionInfo()

maxsize: int = 9223372036854775807
maxunicode: int = 1114111
byteorder: str = "little"
platform: str = "linux"

# No command line reaches the program under verification, so argv holds the
# script name alone, as `python3 main.py` does.
argv: list[str] = [""]

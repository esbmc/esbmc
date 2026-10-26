# pylint: disable=undefined-variable,unused-argument
# `nondet_int` and `nondet_str` are ESBMC intrinsics matched by name by the
# Python converter. No process is started: the child's exit status and output
# are nondeterministic. Captured output is modelled as str whether or not
# text=True is passed; output that is not piped is None, as in CPython.
from typing import Any, Optional

PIPE: int = -1
STDOUT: int = -2
DEVNULL: int = -3


class CalledProcessError(Exception):

    def __init__(self,
                 returncode: int,
                 cmd: Any,
                 output: Optional[str] = None,
                 stderr: Optional[str] = None) -> None:
        self.returncode: int = returncode
        self.cmd: Any = cmd
        self.output: Optional[str] = output
        self.stdout: Optional[str] = output
        self.stderr: Optional[str] = stderr


class CompletedProcess:

    def __init__(self,
                 args: Any,
                 returncode: int,
                 stdout: Optional[str] = None,
                 stderr: Optional[str] = None) -> None:
        self.args: Any = args
        self.returncode: int = returncode
        self.stdout: Optional[str] = stdout
        self.stderr: Optional[str] = stderr

    def check_returncode(self) -> None:
        if self.returncode != 0:
            raise CalledProcessError(self.returncode, self.args, self.stdout, self.stderr)


def run(args: Any,
        stdout: Any = None,
        stderr: Any = None,
        capture_output: bool = False,
        text: bool = False,
        check: bool = False,
        shell: bool = False,
        timeout: Any = None,
        cwd: Any = None) -> CompletedProcess:
    returncode: int = nondet_int()
    out: Optional[str] = None
    err: Optional[str] = None
    if capture_output or stdout == PIPE:
        out = nondet_str()
    if capture_output or stderr == PIPE:
        err = nondet_str()
    if check and returncode != 0:
        raise CalledProcessError(returncode, args, out, err)
    return CompletedProcess(args, returncode, out, err)

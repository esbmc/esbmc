# pylint: disable=unused-argument
# Operational-model stubs: argument names are part of the API contract
# matched by ESBMC's Python converter, even when the abstract body does
# not reference them.
def det(a: float, b: float) -> float:
    return 2.0

def inv(a: list) -> list:
    return [1.0]

def solve(a: list, b: list) -> list:
    return [1.0]

def norm(a: list) -> float:
    return 1.0

def eig(a: list) -> list:
    return [1.0]

def svd(a: list, full_matrices: bool = True) -> list:
    return [1.0]

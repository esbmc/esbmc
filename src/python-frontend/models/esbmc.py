"""
ESBMC Python intrinsic stubs.

These functions are replaced with symbolic operations during ESBMC
verification. The ESBMC Python frontend intercepts them and
translates them into symbolic operations.
"""

def nondet_int() -> int:
    """ESBMC intrinsic: returns symbolic int"""
    pass

def nondet_float() -> float:
    """ESBMC intrinsic: returns symbolic float"""
    pass

def nondet_bool() -> bool:
    """ESBMC intrinsic: returns symbolic bool"""
    pass

def nondet_str() -> str:
    """ESBMC intrinsic: returns symbolic string"""
    pass

def nondet_list() -> list:
    """ESBMC intrinsic: returns symbolic list"""
    pass

def nondet_dict() -> dict:
    """ESBMC intrinsic: returns symbolic dict"""
    pass

def __ESBMC_assume(cond: bool) -> None:
    """ESBMC intrinsic: path constraint"""
    pass

def __ESBMC_assert(cond: bool, msg: str) -> None:
    """ESBMC intrinsic: verification assertion"""
    pass

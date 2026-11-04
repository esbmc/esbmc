"""
Fallback for SV-COMP's _sv_verifier module.

Used only when no _sv_verifier is importable, e.g. on a worker that received
the task's main.py alone. The nondet_* calls are intercepted at their call
sites, so these bodies are placeholders that give each name its signature.
"""


def nondet_bool() -> bool:
    return False


def nondet_int() -> int:
    return 0


def nondet_float() -> float:
    return 0.0


def nondet_str() -> str:
    return ""


def nondet_list(nondet_elem):
    return []


def nondet_dict(nondet_key, nondet_value):
    return {}

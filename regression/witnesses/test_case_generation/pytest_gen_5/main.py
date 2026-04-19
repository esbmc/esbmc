def add_positive(is_p: str, is_q: str) -> int:
    """Add two positive numbers"""
    if is_p == "vip":
        return 0
    elif is_q == "member":
        return 1
    return 0

add_positive(__VERIFIER_nondet_str(), __VERIFIER_nondet_str())
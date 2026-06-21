# Regression for #5499: a `str` *variable* used as a printf format string was
# lowered as pointer modulo and crashed the SMT backend (SIGSEGV). It is now
# rejected with a clean diagnostic instead of crashing. The program itself is
# valid Python (runs cleanly under CPython); only ESBMC cannot model it.
def fmt_int(fmt: str, n: int) -> str:
    return fmt % n


def main() -> None:
    s = fmt_int("%d", 5)
    assert len(s) >= 0


main()

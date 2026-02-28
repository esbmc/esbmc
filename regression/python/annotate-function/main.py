def error() -> None:
    a = b  ## Python annotation will fail here if esbmc is invoked without --function foo


def foo() -> int:
    x = 1
    return x

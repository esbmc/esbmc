def error() -> None:
    a = b  ## Python annotation will fail here if esbmc is invoked without --function foo

def foo() -> int:
    return 1

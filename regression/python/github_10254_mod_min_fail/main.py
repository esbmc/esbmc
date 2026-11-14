# x % -1 is 0, so 100 // 0 raises ZeroDivisionError. In C++, -2**63 % -1
# overflows, and folding it stopped ESBMC with SIGFPE.
def m() -> int:
    x = -9223372036854775807 - 1
    return 100 // (x % -1)


m()

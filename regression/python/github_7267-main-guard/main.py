# Every Python file with a main guard hit this: the guard is folded to a
# constant, so one of the two reachability probes was unsatisfiable by
# construction and reported as CWE-561.
def main():
    assert 1 + 1 == 2


if __name__ == "__main__":
    main()

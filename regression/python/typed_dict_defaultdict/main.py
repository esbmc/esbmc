from collections import defaultdict


def main() -> None:
    n = nondet_int()
    if not 0 <= n <= 2:
        return
    memo = defaultdict(int)
    for i in range(1, n + 1):
        memo[i, 0] = memo[i - 1, 0] + i
    assert memo[n, 0] == n * (n + 1) // 2
    assert memo[n + 5, 7] == 0


main()

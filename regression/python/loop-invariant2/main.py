def main():
    i = 0
    sum = 0

    __loop_invariant(i >= 0 and i <= 5000000 and sum == i * 10)
    while i < 5000000:
        sum += 10
        i += 1

    assert sum == 50000000  # 5000000 * 10

    return 0

main()


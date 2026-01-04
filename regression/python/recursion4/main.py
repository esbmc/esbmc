def sum_even_numbers(n: int) -> int:
    total: int = 0
    i: int = 1
    while i <= n:
        if i % 2 != 0:
            i = i + 1
            continue
        total = total + i
        i = i + 1
    return total

def main() -> None:
    result: int = sum_even_numbers(10)
    assert result == 30

main()

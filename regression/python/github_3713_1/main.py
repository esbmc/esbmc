# Stale mapping: after reassignment from input() to a literal,
# len() must use the actual string length, not the old $input_len$.

def main() -> None:
    x: str = input()
    x = "hello"
    assert len(x) == 5

main()

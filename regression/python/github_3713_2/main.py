# Stale mapping: after reassignment from input() to another variable,
# len() must use strlen on the new value, not the old $input_len$.

def main() -> None:
    x: str = input()
    y: str = "world"
    x = y
    assert len(x) == 5

main()

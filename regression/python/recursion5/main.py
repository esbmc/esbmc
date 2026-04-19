def foo() -> None:
    x: int = 2
    while x < 3:
        x = 3

def main() -> None:
    count: int = 0

    while count < 5:
        count = count + 1

    div: float = 1 / count
    foo()

main()

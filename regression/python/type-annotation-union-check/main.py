from typing import Union


def process(data: Union[int, str]) -> None:
    pass


def main() -> None:
    process(10)
    process("hello")
    process(3.14)


main()

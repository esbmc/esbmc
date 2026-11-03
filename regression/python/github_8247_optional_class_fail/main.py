from typing import Optional


class Ex:

    def shutdown(self) -> None:
        pass


def main() -> None:
    e: Optional[Ex] = None
    if nondet_bool():
        e = Ex()
    e.shutdown()


main()

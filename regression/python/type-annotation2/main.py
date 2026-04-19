class Foo:
    def __init__(self, a: int) -> None:
        self.a = a

    def same(self, b: int) -> bool:
        return self.a == b

def main() -> None:
    f = Foo(1)
    e = f.same(1)    # <----- assign to `e` 

if __name__ == "__main__":
    main()

class MyClass:
    @staticmethod
    def static_func(x: int) -> int:
        return x * 2

MyClass.static_func("not an int")

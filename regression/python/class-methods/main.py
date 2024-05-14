class MyClass:
    @classmethod
    def my_method(cls) -> int:
        return 1

assert MyClass.my_method() == 1

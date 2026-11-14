# Issue #10222: the default of an async method is evaluated when the class body
# runs, so dropping the definition lost its ZeroDivisionError.
class A:

    async def f(self, x: int = 1 // 0) -> None:
        pass

class Other:
    pass


class Schema:

    def describe(self) -> str:
        f = lambda self: self.__class__.__name__
        return f(Other())


assert Schema().describe() == "Other"

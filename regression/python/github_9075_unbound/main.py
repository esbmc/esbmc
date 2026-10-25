class Other:
    pass


class Schema:

    def describe(self) -> str:
        return self.__class__.__name__


assert Schema.describe(Other()) == "Other"

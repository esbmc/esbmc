def make() -> str:

    class Local:

        def name(self) -> str:
            return self.__class__.__name__

    return Local().name()


assert make() == "Local"

class Outer:

    class Inner:
        pass


assert type(Outer.Inner()).__name__ == "Inner"

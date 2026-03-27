# Global variable with nondet value accessed from class method
threshold: int = nondet_int()


class Checker:

    def above_zero(self) -> bool:
        return threshold > 0

    def below_hundred(self) -> bool:
        return threshold < 100


c = Checker()
# Both methods must consistently observe the same global
if c.above_zero():
    assert threshold > 0
if c.below_hundred():
    assert threshold < 100

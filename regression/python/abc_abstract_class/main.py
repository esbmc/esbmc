from abc import ABC, abstractmethod


class S(ABC):
    @abstractmethod
    def area(self):
        pass

    def name(self):
        return "s"


class R(S):
    def area(self):
        return 2


class Mix(R, S):
    pass


assert R().area() == 2
assert R().name() == "s"
assert Mix().area() == 2

try:
    S()
    assert False
except TypeError:
    pass

from abc import ABC, abstractmethod


class S(ABC):
    @abstractmethod
    def area(self):
        pass

    @abstractmethod
    def side(self):
        pass


class Half(S):
    def area(self):
        return 2


h = Half()

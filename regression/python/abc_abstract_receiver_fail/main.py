from abc import ABC, abstractmethod


class S(ABC):
    @abstractmethod
    def area(self):
        pass

    def name(self):
        return "s"


assert S().name() == "s"

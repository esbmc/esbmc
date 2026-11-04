# len(self) in an imported base reaches the subclass's __len__ (#9835).
import lib

class D(lib.C):
    def __len__(self) -> int:
        return 5

assert D().size() == 5

# Test case 14: Deep nested attribute access at Module Level
# Testing the limits of global scope resolution without 'self'

class Level3:
    def get_val(self) -> int:
        return 999

class Level2:
    def __init__(self) -> None:
        self.l3: Level3 = Level3()

class Level1:
    def __init__(self) -> None:
        self.l2: Level2 = Level2()

# Global instantiation
l1 = Level1()

# Deep chain at module level: l1 -> l2 -> l3 -> get_val()
# Chain does NOT start with 'self'.
# No explicit type annotation on 'res' to help the solver.
res = l1.l2.l3.get_val()

assert res == 999
# Test case 6: Four-level nested attribute access
class Level4:
    def get_value(self) -> int:
        return 1234

class Level3:
    def __init__(self) -> None:
        self.level4: Level4 = Level4()

class Level2:
    def __init__(self) -> None:
        self.level3: Level3 = Level3()

class Level1:
    def __init__(self) -> None:
        self.level2: Level2 = Level2()

    def test_four_levels(self) -> int:
        # Test self.level2.level3.level4.get_value() - four levels
        result = self.level2.level3.level4.get_value()
        return result

level1 = Level1()
result: int = level1.test_four_levels()
assert result == 1234


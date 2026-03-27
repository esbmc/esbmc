# Test case 5: Three-level nested attribute access - failing case
class Level3:

    def get_value(self) -> int:
        return 999


class Level2:

    def __init__(self) -> None:
        self.level3: Level3 = Level3()

    def get_level3(self) -> Level3:
        return self.level3


class Level1:

    def __init__(self) -> None:
        self.level2: Level2 = Level2()

    def test_three_levels(self) -> int:
        # Test self.level2.level3.get_value() - three levels
        # No type annotation to trigger type inference
        result = self.level2.level3.get_value()
        return result


level1 = Level1()
result: int = level1.test_three_levels()
# This should fail: expected 999 but assertion is wrong
assert result == 1000

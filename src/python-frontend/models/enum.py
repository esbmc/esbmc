class Enum:
    def __init__(self, value: int, name: str):
        self.value: int = value
        self.name: str = name

    def __eq__(self, other: Enum) -> bool:
        return self.value == other.value

    def __ne__(self, other: Enum) -> bool:
        return self.value != other.value

    def __hash__(self) -> int:
        return self.value

    def __repr__(self) -> str:
        return self.name

    def __str__(self) -> str:
        return self.name
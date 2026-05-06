# pylint: disable=undefined-variable
# `Enum` self-references in method signatures are forward declarations
# resolved by ESBMC's Python converter; they have no Python binding.

class Enum:

    def __init__(self, value: int, name: str):
        self.value: int = value
        self.name: str = name

    def __eq__(self, other: Enum) -> bool:
        """Return True if both enum members share the same value."""
        return self.value == other.value

    def __ne__(self, other: Enum) -> bool:
        """Return True if the enum members have different values."""
        return self.value != other.value

    def __hash__(self) -> int:
        return self.value

    def __repr__(self) -> str:
        return self.name

    def __str__(self) -> str:
        return self.name

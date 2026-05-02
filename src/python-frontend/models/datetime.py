# Operational model for datetime module
# pylint: disable=function-redefined  # intentional stdlib shadow for ESBMC models


class datetime:
    """Represent a date and time."""

    def __init__(self, year: int, month: int, day: int):
        """Initialize a datetime with the given calendar date; time fields default to 0."""
        self.year: int = year
        self.month: int = month
        self.day: int = day
        self.hour: int = 0
        self.minute: int = 0
        self.second: int = 0
        self.microsecond: int = 0

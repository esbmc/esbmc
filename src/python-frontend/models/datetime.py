# Operational model for datetime module

class datetime:
    """Represents a date and time"""

    def __init__(self,
                 year: int,
                 month: int,
                 day: int):
        self.year: int = year
        self.month: int = month
        self.day: int = day
        self.hour: int = 0
        self.minute: int = 0
        self.second: int = 0
        self.microsecond: int = 0
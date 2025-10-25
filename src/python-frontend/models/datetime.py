# Operational model for datetime module

class datetime:
    """Represents a date and time"""
    
    def __init__(self, year: int, month: int, day: int, 
                 hour: int = 0, minute: int = 0, second: int = 0, microsecond: int = 0) -> None:
        self.year: int = year
        self.month: int = month
        self.day: int = day
        self.hour: int = hour
        self.minute: int = minute
        self.second: int = second
        self.microsecond: int = microsecond
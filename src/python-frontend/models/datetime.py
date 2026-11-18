# Operational model for datetime module
# pylint: disable=function-redefined,unused-wildcard-import  # intentional stdlib shadow for ESBMC models
# pylint: disable=too-many-arguments,too-many-positional-arguments,unused-argument

from typing import Optional


class datetime:
    """Represent a date and time."""

    def __init__(self,
                 year: int,
                 month: int,
                 day: int,
                 hour: int = 0,
                 minute: int = 0,
                 second: int = 0,
                 microsecond: int = 0):
        """Initialize a naive datetime; time fields default to 0."""
        self.year: int = year
        self.month: int = month
        self.day: int = day
        self.hour: int = hour
        self.minute: int = minute
        self.second: int = second
        self.microsecond: int = microsecond

    # Only naive datetimes are modelled, so tzinfo can only be None.
    def replace(self,
                year: Optional[int] = None,
                month: Optional[int] = None,
                day: Optional[int] = None,
                hour: Optional[int] = None,
                minute: Optional[int] = None,
                second: Optional[int] = None,
                microsecond: Optional[int] = None,
                tzinfo: None = None) -> "datetime":  # pylint: disable=redefined-outer-name
        """Return a copy with the given fields replaced."""
        if year is None:
            year = self.year
        if month is None:
            month = self.month
        if day is None:
            day = self.day
        if hour is None:
            hour = self.hour
        if minute is None:
            minute = self.minute
        if second is None:
            second = self.second
        if microsecond is None:
            microsecond = self.microsecond
        return datetime(year, month, day, hour, minute, second, microsecond)

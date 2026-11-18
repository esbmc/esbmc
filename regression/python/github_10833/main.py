from datetime import datetime

date: datetime = datetime(2024, 3, 15)
naive: datetime = date.replace(tzinfo=None)
assert naive.year == 2024 and naive.month == 3 and naive.day == 15

moved: datetime = date.replace(year=2020, day=1)
assert moved.year == 2020 and moved.month == 3 and moved.day == 1
assert date.year == 2024 and date.day == 15

timed: datetime = moved.replace(hour=5, minute=6)
assert timed.hour == 5 and timed.minute == 6 and timed.second == 0
assert timed.year == 2020

positional: datetime = date.replace(2001)
assert positional.year == 2001 and positional.month == 3

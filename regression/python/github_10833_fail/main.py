from datetime import datetime

date: datetime = datetime(2024, 1, 1)
result: datetime = date.replace(tzinfo=None, month=2)
assert result.month == 1

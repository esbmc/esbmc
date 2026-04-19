total_seconds = 3665
hours, remainder = divmod(total_seconds, 3600)
minutes, seconds = divmod(remainder, 60)

assert hours == 1
assert minutes == 1
assert seconds == 5

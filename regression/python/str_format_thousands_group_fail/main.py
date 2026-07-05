# 1000 grouped is "1,000", not "1000".
assert "{:,}".format(1000) == "1000"

# #9404: `str % None` formats None with str(), it does not raise TypeError.
try:
    s = "%s" % None
except TypeError:
    s = "caught"
assert s == "caught"

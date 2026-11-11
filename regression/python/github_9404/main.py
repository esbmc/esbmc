# #9404: `str % None` formats None with str(), it does not raise TypeError.
s = "%s" % None
assert s == "None"
t = "<%s|%s>" % (None, 1)
assert t == "<None|1>"

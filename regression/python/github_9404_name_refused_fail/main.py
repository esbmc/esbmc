# #9404: a None held in a variable formats as "None", not as the empty
# string a made-up result would give.
x = None
s = "%s" % x
assert s == ""

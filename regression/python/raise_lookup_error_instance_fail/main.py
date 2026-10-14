# An exception instance held in a variable keeps its type when raised.
x = LookupError("missing")
raise x

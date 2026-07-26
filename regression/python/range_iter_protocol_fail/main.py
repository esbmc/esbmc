# Consuming past the end of the sequence raises StopIteration; left uncaught it
# is a reachable failure, confirming the iterator length is modelled (not vacuous).
r = range(2)
it = r.__iter__()
a = next(it)
b = next(it)
c = next(it)  # StopIteration

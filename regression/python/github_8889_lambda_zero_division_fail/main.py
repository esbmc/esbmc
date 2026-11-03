# Issue #8889: a division by zero in a lambda body raises ZeroDivisionError.
f = lambda x: 10 / x
r = f(0)

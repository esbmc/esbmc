foo = nondet_str()

if foo == "":
    foo = "foo"

assert foo != ""

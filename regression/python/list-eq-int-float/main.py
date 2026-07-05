# Mixed int/float list equality follows Python numeric semantics: 1 == 1.0.
assert [1, 2] == [1.0, 2.0]
assert [1.0, 2.0] == [1, 2]
assert [1] != [1.5]
assert ([1] == [1.5]) == False

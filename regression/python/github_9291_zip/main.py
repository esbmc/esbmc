names = ["a", "b", "c"]
ages = nondet_list(3)
result = list(zip(names, ages))
for name, age in result:
    if not isinstance(age, int):
        assert False

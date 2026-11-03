ages = nondet_list(3)
for i, age in enumerate(ages):
    if not isinstance(age, int):
        assert False

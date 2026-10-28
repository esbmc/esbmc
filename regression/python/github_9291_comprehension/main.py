ages = nondet_list(3)
result = [(k, ages[k]) for k in range(len(ages))]
for i, (k, age) in enumerate(result):
    if not isinstance(age + 1, int):
        assert False

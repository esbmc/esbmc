def sum_of_positive_elements(x: dict[int, int], y: list) -> None:
    if 2 in x:
        all_positive = True
        if 3 in x:
            all_positive = False
        if all_positive:
            total = 0
            if total in x:
                total += 1


x = nondet_dict(2, key_type=nondet_int(), value_type=nondet_int())
sum_of_positive_elements(x, "2")
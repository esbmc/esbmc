def sum_of_positive_elements(x: list[int]) -> None:
    if len(x) > 0:
        all_positive = True
        for elem in x:
            if elem <= 0:
                all_positive = False
                break
        
        if all_positive:
            total = 0
            for elem in x:
                total += elem
            assert total > 0

x = nondet_list()
sum_of_positive_elements(x)
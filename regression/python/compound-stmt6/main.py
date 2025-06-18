def test_inside_loop_and_condition():
    total = 0
    for i in range(3):
        total += i
        if i == 1:
            total *= 2
    assert total == 4

test_inside_loop_and_condition()

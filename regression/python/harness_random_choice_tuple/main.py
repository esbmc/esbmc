# A tuple has no model variant: its arity and member types vary per call site,
# so function_call_expr::fold_random_choice_over_tuple selects an element
# inline. The members must still share one type -- see random_choice_none_tuple
# for the case that does not.
# ENSURES: E1 - an int tuple yields one of its members.
#          E2 - a float tuple yields one of its members.
import random


def main() -> None:
    a = random.choice((10, 20, 30))
    assert a == 10 or a == 20 or a == 30  # E1
    b = random.choice((1.5, 2.5))
    assert b == 1.5 or b == 2.5  # E2


main()

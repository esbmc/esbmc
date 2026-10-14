def main() -> None:
    # `in` / `not in` on a dict comprehension result previously raised
    # "Unsupported expression for 'in' operation": membership keyed off the
    # dict's aggregate-kind marker, which is dropped as the comprehension result
    # flows through type inference, while the __python_dict__ tag survives.
    d = {x: x * x for x in range(3)}
    assert 2 in d
    assert 5 not in d

    # Dict literal membership is unchanged.
    f = {1: 10}
    assert 1 in f and 3 not in f


main()

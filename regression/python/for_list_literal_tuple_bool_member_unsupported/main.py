def run():
    first = 0
    # A bool member mismatches the tuple AST in the solver, so the literal
    # keeps no tuple annotation and the unpack is refused.
    for flagged in [(True, 1), (False, 2)]:
        a, b = flagged
        if first == 0:
            assert a
            assert b == 1
        first = first + 1
    assert first == 2


run()

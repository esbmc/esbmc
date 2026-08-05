# Two comparison shapes that reached the solver unreconciled under
# --python-irep2-adjust-only: `x is True` builds floatbv == bool (bitwuzla
# mk_eq sort-width abort), and a chained comparison against integer bounds
# builds lessthanequal over floatbv and signedbv (convert_ast_node signedbv
# assertion).
is_even = lambda x: x % 2 == 0
assert is_even(4) is True
assert is_even(5) is False

between = lambda x: 0 <= x <= 10
assert between(5) is True
assert between(-1) is True

# github #8090: a comparison reaching __ESBMC_list_eq gave no verdict without
# --unwind, because symex never folded reads of the list's element storage.


def same(x: list) -> bool:
    return x == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


m: list = ["y"]
assert m == ["y"]

n: list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
assert same(n)

w: list = ["ab", "cd", "ef"]
assert w == ["ab", "cd", "eg"]

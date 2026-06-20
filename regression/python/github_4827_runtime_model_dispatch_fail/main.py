# Negative variant of github_4827_runtime_model_dispatch: one assertion is
# deliberately wrong, so the runtime-model dispatch must produce the real value
# and refute it. Guards against the dispatch silently returning nondet (which
# would not deterministically fail) -- the soundness check behind #4827.


def f_swapcase(s: str) -> str:
    return s.swapcase()


def f_count(s: str, c: str) -> int:
    return s.count(c)


if __name__ == "__main__":
    assert f_swapcase("Hello") == "hELLO"
    assert f_count("aabba", "a") == 99   # wrong: real count is 3

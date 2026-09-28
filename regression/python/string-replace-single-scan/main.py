def foo(x: str) -> None:
    # count exhausted before the end: the scan stops early and everything
    # past that point must be copied verbatim.
    assert x.replace("aa", "x", 1) == "x-bb-aa"
    assert x.replace("aa", "yy") == "yy-bb-yy"
    assert x.replace("aa", "") == "-bb-"
    # no match at all: the scan marks every position it steps over as a miss.
    assert x.replace("zz", "q") == "aa-bb-aa"
    # a needle that only fits in the tail the scan cannot reach.
    assert x.replace("a-", "Q") == "aQbb-aa"


foo("aa-bb-aa")

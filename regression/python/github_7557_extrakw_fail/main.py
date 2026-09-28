# A keyword the rewrite cannot place must leave the whole call alone. Binding
# only `object` here would drop `encoding` silently -- the #7557 failure mode,
# reintroduced by the rewrite itself. CPython raises TypeError for this call.
def main() -> None:
    x = str(object=500, encoding="utf-8")
    assert x == "500"


main()

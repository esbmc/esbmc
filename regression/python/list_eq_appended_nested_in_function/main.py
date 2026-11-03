# As list_eq_appended_nested, inside a function, where the locals the type
# map names are created as file-local symbols like the frontend's own.
def main() -> None:
    ll: list[list[int]] = [[5], [6]]
    ret: list[list[int]] = []
    for rest in ll:
        ret.append(rest)
    assert ret == [[5], [6]]


main()

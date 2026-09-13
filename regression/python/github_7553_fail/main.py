# The keys view equals {1}, so != is false. This was proved before the view
# was made set-like, whatever the contents (#7553).
def main() -> None:
    assert {1: 1}.keys() != {1}


main()

# A non-finite float literal reaches the frontend with a null value; its !r
# is not "None" (#7559).
def main() -> None:
    assert f'{1e999!r}' != 'None'


main()

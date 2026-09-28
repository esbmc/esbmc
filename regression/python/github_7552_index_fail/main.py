# ord() of an indexed character sign-extended its byte, proving it negative
# (#7552).
def main() -> None:
    s: str = "éa"
    i: int = 0
    assert ord(s[i]) < 0


main()

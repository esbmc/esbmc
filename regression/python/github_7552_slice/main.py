# Byte-wise slicing leaves a lead byte without its continuation bytes; ord()
# must not read past the string decoding it (#7552).
def main() -> None:
    n: int = 65536
    t = chr(n)[0:1]
    x = ord(t)
    s: str = "\U0001F600"
    y = ord(s[0])


main()

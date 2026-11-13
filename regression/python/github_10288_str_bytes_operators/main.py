def raises(op: str) -> bool:
    try:
        if op == "&":
            "6" & "3"
        elif op == "|":
            "6" | "3"
        elif op == "^":
            "6" ^ "3"
        elif op == "@":
            "6" @ "3"
        elif op == "<<":
            "6" << "3"
        elif op == ">>":
            "6" >> "3"
        elif op == "b/":
            b"ab" / b"cd"
        elif op == "b-":
            b"ab" - b"cd"
        elif op == "b&":
            b"ab" & b"cd"
        else:
            "6" & 3
    except TypeError:
        return True
    return False


for op in ["&", "|", "^", "@", "<<", ">>", "b/", "b-", "b&", "x"]:
    assert raises(op)
assert "6" + "3" == "63"
assert b"ab" + b"cd" == b"abcd"

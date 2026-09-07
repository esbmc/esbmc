# A locally-caught raise must not anchor the type. The ValueError on line 6
# is handled where it is raised, so the property belongs to the escaping
# raise on line 12 -- before the fix the caught line won the anchor.
def caught() -> None:
    try:
        raise ValueError("handled here")
    except ValueError:
        pass


def escaping() -> None:
    raise ValueError("escapes")


def main() -> None:
    caught()
    escaping()


main()

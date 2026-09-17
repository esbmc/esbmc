# A **kwargs splat carries no statically known names, so the rewrite must leave
# the call alone: binding the splat to a positional slot turns int(**d) into
# int(d) and crashes the converter (#7557).
def main() -> None:
    d: dict = {}
    assert int(**d) == 0


main()

# The frontend folds an assert its constant evaluator can prove, which drops
# the call inside it. Everything reachable only through that call is then never
# symexed, and --dead-code-check reports all of it as CWE-561. Under that flag
# the call has to survive: reachability is the thing being measured.
def pick(x: int) -> int:
    if x > 0:
        return 1
    if x > 0:
        return 2
    return 0


def main():
    assert pick(5) == 1


main()

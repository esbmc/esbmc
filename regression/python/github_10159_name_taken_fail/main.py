# The renamed def must not take a name the module already binds (#10159).
def len(v) -> int:
    return 3


def len__esbmc_user(v) -> int:
    return 99


assert len([1]) == 99

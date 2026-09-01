class uint64(int):
    pass


class Slot(uint64):
    pass


class GeneralizedIndex(int):
    pass


FINALIZED_ROOT_GINDEX = GeneralizedIndex(105)


def finalized_root_gindex_at_slot(slot: Slot) -> GeneralizedIndex:
    # pylint: disable=unused-argument
    return FINALIZED_ROOT_GINDEX


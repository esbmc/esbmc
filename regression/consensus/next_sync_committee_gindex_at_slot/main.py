class uint64(int):
    pass


class Slot(uint64):
    pass


class GeneralizedIndex(int):
    pass


NEXT_SYNC_COMMITTEE_GINDEX = GeneralizedIndex(55)


def next_sync_committee_gindex_at_slot(slot: Slot) -> GeneralizedIndex:
    # pylint: disable=unused-argument
    return NEXT_SYNC_COMMITTEE_GINDEX


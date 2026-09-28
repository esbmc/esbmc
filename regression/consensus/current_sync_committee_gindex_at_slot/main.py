class uint64(int):
    pass


class Slot(uint64):
    pass


class GeneralizedIndex(int):
    pass


CURRENT_SYNC_COMMITTEE_GINDEX = GeneralizedIndex(54)


def current_sync_committee_gindex_at_slot(slot: Slot) -> GeneralizedIndex:
    # pylint: disable=unused-argument
    return CURRENT_SYNC_COMMITTEE_GINDEX


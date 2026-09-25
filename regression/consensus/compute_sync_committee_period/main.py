class uint64(int):
    pass


class Epoch(uint64):
    pass


EPOCHS_PER_SYNC_COMMITTEE_PERIOD = uint64(256)


def compute_sync_committee_period(epoch: Epoch) -> uint64:
    return epoch // EPOCHS_PER_SYNC_COMMITTEE_PERIOD


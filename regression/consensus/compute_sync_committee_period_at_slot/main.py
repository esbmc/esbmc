class uint64(int):
    pass


class Slot(uint64):
    pass


class Epoch(uint64):
    pass


EPOCHS_PER_SYNC_COMMITTEE_PERIOD = uint64(256)


def compute_sync_committee_period(epoch: Epoch) -> uint64:
    return epoch // EPOCHS_PER_SYNC_COMMITTEE_PERIOD


SLOTS_PER_EPOCH = uint64(32)


def compute_epoch_at_slot(slot: Slot) -> Epoch:
    """
    Return the epoch number at ``slot``.
    """
    return Epoch(slot // SLOTS_PER_EPOCH)


def compute_sync_committee_period_at_slot(slot: Slot) -> uint64:
    return compute_sync_committee_period(compute_epoch_at_slot(slot))


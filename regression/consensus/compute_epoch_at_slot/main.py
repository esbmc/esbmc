class uint64(int):
    pass


class Slot(uint64):
    pass


class Epoch(uint64):
    pass


SLOTS_PER_EPOCH = uint64(32)


def compute_epoch_at_slot(slot: Slot) -> Epoch:
    """
    Return the epoch number at ``slot``.
    """
    return Epoch(slot // SLOTS_PER_EPOCH)


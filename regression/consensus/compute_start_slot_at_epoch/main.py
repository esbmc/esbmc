class uint64(int):
    pass


class Epoch(uint64):
    pass


class Slot(uint64):
    pass


SLOTS_PER_EPOCH = uint64(32)


def compute_start_slot_at_epoch(epoch: Epoch) -> Slot:
    """
    Return the start slot of ``epoch``.
    """
    return Slot(epoch * SLOTS_PER_EPOCH)


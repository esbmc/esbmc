class uint64(int):
    pass


class Slot(uint64):
    pass


SLOTS_PER_EPOCH = uint64(32)


def is_shuffling_stable(slot: Slot) -> bool:
    return slot % SLOTS_PER_EPOCH != 0


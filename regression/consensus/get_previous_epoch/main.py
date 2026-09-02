class uint64(int):
    pass


Root = bytes


class Slot(uint64):
    pass


class BeaconState:
    genesis_time: uint64
    genesis_validators_root: Root
    slot: Slot


class Epoch(uint64):
    pass


GENESIS_EPOCH = Epoch(0)


SLOTS_PER_EPOCH = uint64(32)


def compute_epoch_at_slot(slot: Slot) -> Epoch:
    """
    Return the epoch number at ``slot``.
    """
    return Epoch(slot // SLOTS_PER_EPOCH)


def get_current_epoch(state: BeaconState) -> Epoch:
    """
    Return the current epoch.
    """
    return compute_epoch_at_slot(state.slot)


def get_previous_epoch(state: BeaconState) -> Epoch:
    """
    Return the previous epoch (unless the current epoch is ``GENESIS_EPOCH``).
    """
    current_epoch = get_current_epoch(state)
    return GENESIS_EPOCH if current_epoch == GENESIS_EPOCH else Epoch(current_epoch - 1)


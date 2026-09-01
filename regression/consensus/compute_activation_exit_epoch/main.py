class uint64(int):
    pass


class Epoch(uint64):
    pass


MAX_SEED_LOOKAHEAD = uint64(4)


def compute_activation_exit_epoch(epoch: Epoch) -> Epoch:
    """
    Return the epoch during which validator activations and exits initiated in ``epoch`` take effect.
    """
    return Epoch(epoch + 1 + MAX_SEED_LOOKAHEAD)


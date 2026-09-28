Bytes32 = bytes


class uint64(int):
    pass


class Gwei(uint64):
    pass


class Epoch(uint64):
    pass


class Validator:#(Container):
#    pubkey: BLSPubkey
    withdrawal_credentials: Bytes32  # Commitment to pubkey for withdrawals
    effective_balance: Gwei  # Balance at stake
#    slashed: boolean
    slashed: bool
    # Status epochs
    activation_eligibility_epoch: Epoch  # When criteria for activation were met
    activation_epoch: Epoch
    exit_epoch: Epoch
    withdrawable_epoch: Epoch  # When validator can withdraw funds


FAR_FUTURE_EPOCH = Epoch(2**64 - 1)


MAX_EFFECTIVE_BALANCE = Gwei(32000000000)


def is_eligible_for_activation_queue(validator: Validator) -> bool:
    """
    Check if ``validator`` is eligible to be placed into the activation queue.
    """
    return (
        validator.activation_eligibility_epoch == FAR_FUTURE_EPOCH
        and validator.effective_balance == MAX_EFFECTIVE_BALANCE
    )


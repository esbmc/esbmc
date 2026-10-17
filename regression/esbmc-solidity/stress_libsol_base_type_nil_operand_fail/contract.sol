// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

// Dual to pass: same library pattern, but go() unconditionally fails
// with assert(false). Without the base_type nil-guard fix the frontend
// SIGSEGVs inside goto_check before symex ever runs; with the fix the
// frontend reaches symex, which then trivially reports the seeded
// assert(false) as VERIFICATION FAILED.

library FarmAcc {
    uint256 internal constant _SCALE = 1e18;
    struct Info {
        uint40 finished;
        uint32 duration;
        uint184 reward;
        uint256 balance;
    }
    function farmedSinceCheckpointScaled(Info storage info, uint256 checkpoint)
        internal view returns (uint256 amount)
    {
        unchecked {
            (uint40 finished, uint32 duration, uint184 reward) =
                (info.finished, info.duration, info.reward);
            if (duration > 0) {
                uint256 elapsed =
                    (block.timestamp < finished ? block.timestamp : finished) -
                    (checkpoint < finished ? checkpoint : finished);
                return elapsed * reward * _SCALE / duration;
            }
        }
    }
}

library UserAcc {
    struct Info {
        mapping(address => int256) corrections;
    }
    function farmed(Info storage info, address account, uint256 balance, uint256 fpt)
        internal view returns (uint256)
    {
        return uint256(int256(balance * fpt) - info.corrections[account]) / FarmAcc._SCALE;
    }
}

contract C {
    FarmAcc.Info private farmInfo;
    UserAcc.Info private userInfo;

    function go() external pure {
        assert(false);
    }
}

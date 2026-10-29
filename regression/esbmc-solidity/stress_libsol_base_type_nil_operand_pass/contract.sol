// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

// Regression for a SIGSEGV in `base_type(expr2tc&)` during goto_check's
// div_by_zero_check on a library function containing
// `uint256(int256(balance * fpt) - info.corrections[account]) / _SCALE`
// alongside `farmedSinceCheckpointScaled`. The frontend emits a nil
// sub-operand somewhere inside the div's divisor subtree, and
// base_type's recursive Foreach_operand walk used to deref the nil
// instead of skipping it. Added a nil-expression guard at the top of
// base_type(expr2tc&) to match the way check_rec already treats nil
// sub-exprs.
//
// Extracted from 1inch/farming's FarmAccounting + UserAccounting
// libraries via FarmingHook / FarmingPool / MultiFarmingHook. The mere
// presence of both library functions triggers the crash — they do not
// need to be reachable from the contract's external entry points,
// because goto_check walks the entire goto_functions map.

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

    // go() is intentionally a trivial pure assert so the pass test does
    // not collide with the library functions' real div-by-zero VCCs;
    // the frontend still lowers the full library bodies, and goto_check
    // still walks them, so the crash path is exercised regardless.
    function go() external pure {
        assert(true);
    }
}

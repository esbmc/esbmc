// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.8.0;

// The converter emits >>= without picking the shift kind; remove_sideeffects
// handles only the resolved spellings, so an unrewritten assign_shr aborts.
// Each operand has its top bit set, where the two kinds disagree.
contract C {
    function f() public pure {
        uint x = 1 << 255;
        x >>= 255;
        assert(x == 0);
        int y = -8;
        y >>= 1;
        assert(y == -4);
    }
}

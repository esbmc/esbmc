// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.8.0;

// The converter emits >>= without picking the shift kind; remove_sideeffects
// handles only the resolved spellings, so an unrewritten assign_shr aborts.
contract C {
    function f() public pure {
        uint x = 8;
        x >>= 2;
        assert(x == 3);
    }
}

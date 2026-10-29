// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.8.0;

// Control for finding #2: a scalar state variable lives directly in the
// `*this` struct, so the existing `*this = _sol_save_this` rollback DOES
// restore it across a revert.  This must stay SUCCESSFUL after the
// global-store rollback fix (guards the fix from over-reaching).
contract C {
    uint s;

    function setAndRevert() public {
        s = 1;
        revert();
    }

    function checkClean() public view {
        assert(s == 0);
    }
}

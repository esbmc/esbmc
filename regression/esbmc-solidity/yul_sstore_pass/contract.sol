// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

contract H {
    uint256 x;

    // Confirms `sstore(x.slot, v)` is precise: writes v into the
    // state variable x. Under the legacy havoc fallback the assembly
    // block re-nondets x and the assertion would FAIL.
    function check(uint256 v) public {
        assembly {
            sstore(x.slot, v)
        }
        assert(x == v);
    }
}

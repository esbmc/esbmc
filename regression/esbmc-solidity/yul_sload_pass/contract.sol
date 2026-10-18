// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

contract H {
    uint256 x;

    // Confirms `sload(x.slot)` is precise: reads the actual value of
    // state variable x rather than nondet. Under the legacy havoc
    // fallback the assembly block re-nondets x via its `.slot` ext-ref
    // and r could be anything, so the assertion would FAIL.
    function check(uint256 v) public {
        x = v;
        uint256 r;
        assembly {
            r := sload(x.slot)
        }
        assert(r == v);
    }
}

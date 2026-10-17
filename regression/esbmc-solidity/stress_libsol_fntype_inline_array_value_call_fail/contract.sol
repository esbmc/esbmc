// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

// Negative test: [add1, add2][0](val) resolves to add1(val) = val+1,
// but the assertion expects val+2 — must produce VERIFICATION FAILED.
contract C {
    function add1(uint x) internal pure returns (uint) { return x + 1; }
    function add2(uint x) internal pure returns (uint) { return x + 2; }

    function check(uint val) public pure {
        uint result = [add1, add2][0](val);
        assert(result == val + 2); // WRONG: should be val+1
    }
}

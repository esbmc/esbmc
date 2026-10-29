// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

// Inline array of internal function pointers with constant index.
// [add1, add2][0](val) must constant-fold to add1(val) and return val+1.
// [add1, add2][1](val) must constant-fold to add2(val) and return val+2.
contract C {
    function add1(uint x) internal pure returns (uint) { return x + 1; }
    function add2(uint x) internal pure returns (uint) { return x + 2; }

    function check(uint val) public pure {
        uint r0 = [add1, add2][0](val);
        uint r1 = [add1, add2][1](val);
        assert(r0 == val + 1);
        assert(r1 == val + 2);
    }
}

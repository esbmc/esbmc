// SPDX-License-Identifier: GPL-3.0
pragma solidity ^0.8.0;

// Regression: a bytesN return type is modeled as a BytesStatic struct.
// `return 0;` (an integer literal) must be lowered to a real BytesStatic
// value, not a raw typecast(int -> struct) that crashes the value-set
// analysis (value_sett::make_member abort / Release SIGSEGV).
contract C {
    function f() public pure returns (bytes32) { return 0; }
    function check() public pure {
        bytes32 x = f();
        assert(x == bytes32(0));
    }
}

// SPDX-License-Identifier: MIT
// Regression for Block 2+3: a user-written narrowing cast
// `uint8(uint256value)` must be instrumented by `cast_overflow_check`
// by DEFAULT. Previously narrowing was gated on
// --overflow-check / --unsigned-overflow-check; after the decouple,
// the narrowing check has its own toggle and is on by default as part
// of the standard check set.
pragma solidity >=0.8.0;

contract T {
    function narrow(uint256 x) public pure returns (uint8) {
        return uint8(x);
    }
}

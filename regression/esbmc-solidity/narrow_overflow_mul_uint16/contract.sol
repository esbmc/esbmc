// SPDX-License-Identifier: GPL-3.0
pragma solidity ^0.8.0;

// MULTIPLICATION AT uint16 -- a different OPERATOR and a different WIDTH from
// the two tests that pin uint8 addition.
//
// The promotion this pins was per-operand, not per-operator, so a suite that
// only covered `+` at uint8 would have gone green on a fix that handled one
// operator. uint16 also sits on the other side of `get_c_type`'s UCHAR/UINT
// split from uint8, so the two widths reach the promotion by different routes
// and only one of them is exercised by the addition tests.
//
// Measured on the defective build: VERIFICATION SUCCESSFUL. On chain
// 300 * 300 = 90000 does not fit in uint16 and reverts Panic(0x11).
contract NarrowMul {
    function test_mul_overflow() public pure returns (uint16) {
        uint16 x = 300;
        uint16 y = x * 300; // 90000 > type(uint16).max
        return y;
    }
}

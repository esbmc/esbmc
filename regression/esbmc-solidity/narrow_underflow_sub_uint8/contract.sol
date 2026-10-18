// SPDX-License-Identifier: GPL-3.0
pragma solidity ^0.8.0;

// UNSIGNED UNDERFLOW AT A NARROW WIDTH.
//
// `int_boundary_2` and `unchecked_block_4` already pin uint8 ADDITION. This one
// exists because subtraction is the shape the defect hid MOST cleanly and is the
// commonest real Solidity panic: the operands used to be promoted to a 32-bit
// SIGNED int before the overflow claim was built, and `0 - 1` is a perfectly
// ordinary -1 in int32, so the claim was true for every model.
//
// Measured on the defective build: VERIFICATION SUCCESSFUL. On chain this
// reverts Panic(0x11).
contract NarrowUnderflow {
    function test_underflow() public pure returns (uint8) {
        uint8 x = 0;
        uint8 y = x - 1; // underflow in Solidity 0.8+
        return y;
    }
}

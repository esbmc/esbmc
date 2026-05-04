// SPDX-License-Identifier: GPL-3.0
pragma solidity ^0.8.0;

// Test that narrowing cast loses data — assertion should fail
contract TypeConvFail {
    function test_narrow_fail() public pure {
        uint256 val = 256;
        uint8 narrow = uint8(val); // 256 mod 256 = 0
        // Wrong assertion: narrowed value is 0, not 256
        assert(narrow == 256);
    }
}

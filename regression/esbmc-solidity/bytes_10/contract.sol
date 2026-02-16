// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract Test {
    function test() public pure {
        uint8 x = 0x12;
        uint8 y = 0x0012;
        assert(x==y);
        address z = 0x0034567890123456789012345678901234567890;
        assert(bytes20(z)[0] == 0x00);
    }
}

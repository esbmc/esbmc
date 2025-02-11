// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.5.0;

contract BASE {
    function test() public {
        bytes memory b;
        b = new bytes(7);
        assert(b[0] == 0x00);
        
        bytes1[] memory x;
        x = new bytes1[](10);
        x[0] = 0x10;
        assert(x[0] == 0x10);
    }
}

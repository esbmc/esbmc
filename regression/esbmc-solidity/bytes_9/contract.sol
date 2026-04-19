// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract NewBytesExample {
    function createBytes() public pure returns (bytes memory) {
        bytes memory b = new bytes(4);
        b[0] = 0x11;
        b[1] = 0x22;
        b[2] = 0x33;
        b[3] = 0x44;
        assert(b[0] ==  (new bytes(4))[0]);
    }
}

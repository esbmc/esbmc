// SPDX-License-Identifier: GPL-3.0
pragma solidity ^0.8.17;

contract Base {
    function Conv() public {
    bytes memory a = hex"3423";

    uint32 c = uint32(bytes4(a)); 
    assert(c == 0x34230000);

    }
}

// SPDX-License-Identifier: GPL-3.0
pragma solidity ^0.8.17;

contract Base {
    function Conv() public {
    bytes2 a = 0x3423;

    uint32 c = uint32(bytes4(a)); // c will be 0x34230000
    uint8 d = uint8(uint16(a)); // d will be 0x23
    uint8 e = uint8(bytes1(a)); // e will be 0x34
    
    assert(c == 0x34230000);
    assert(d == 0x23);
    assert(e == 0x34);
    }
}

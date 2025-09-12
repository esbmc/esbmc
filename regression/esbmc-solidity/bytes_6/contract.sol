// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract ArrayAndBytesBuiltinTest {
    uint x = 0x12;
    string s = "12";
    bytes a = bytes("1234");
    bytes b = bytes(hex"1234");
    bytes2 c = bytes2(hex"1234");
    bytes2 d = bytes2(0x1234);
    bytes e = bytes(s);
    bytes f = hex"1234";
    bytes g = "0x1234";
}

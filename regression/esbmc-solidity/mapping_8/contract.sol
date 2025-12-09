// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.5.0;

contract Base {
    mapping(bytes4 => bytes4) test1;

    constructor() {
        data1 = "test";
        data2 = 0x74657374; // "test"

        test1[data2];
        assert(test1[data2] == 0);
        test1[data2] = bytes4(data1);
        assert(test1[data2] != 0x74657374);
    }
    bytes16 data1;
    bytes4 data2;
}

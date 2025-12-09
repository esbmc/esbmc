// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.5.0;

contract Base {
    enum T {
        xxx,
        yyy
    }

    mapping(bytes => bytes) test1;
    mapping(address => address) test2;
    mapping(bool => bool) test3;
    mapping(int => T) test4;

    bytes data1;
    bytes data2;

    address x = 0x1f9840a85d5aF5bf1D1762F925BDADdC4201F984;
    address y = 0x1f9840a85d5aF5bf1D1762F925BDADdC4201F984;

    constructor() {
        data1 = "t";
        data2 = "t";
        test1[data1];
        test1[data1] = data2;

        assert(test2[x] == test2[y]);

        test3[true] == false;
        bool z = true;
        test3[z] = true;
        assert(test3[true] == true);

        test4[-1] = T.yyy;
        assert(test4[-1] != T.yyy);
    }
}

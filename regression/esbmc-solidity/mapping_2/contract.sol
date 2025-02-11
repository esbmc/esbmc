// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.5.0;

contract Base {
    mapping(string => int8) test2;

    int8 x = 1;
    int8 y = 10;
    string z = "1";

    constructor() {
        test2[z] = x;

        assert(test2[z] == y);
    }
}

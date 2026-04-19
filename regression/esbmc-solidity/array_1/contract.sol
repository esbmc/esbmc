// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.8.0;

uint constant x = 1;
uint constant y = x;

contract Base {
    int[y] arr;
    constructor() {
        assert(arr[y] == 0);
    }
}

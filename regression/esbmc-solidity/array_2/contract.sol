// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.8.0;

uint constant x = 1;
uint constant y = x;

contract Base {
    int[y] arr;
    int[] dynarr;
    constructor() {
        int[y] memory arr2;
        arr2 = arr;
        assert(arr2[0] == 0);
        assert(arr2.length == y);
        assert(dynarr.length == 1);
    }
}

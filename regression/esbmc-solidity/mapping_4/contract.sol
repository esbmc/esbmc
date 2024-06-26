// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.5.0;

contract Base {
    function test() public {
        string memory s = "1";
        test2[s] = 1;
        assert(test2[s] == 0);
    }
    mapping(string => uint) test2;
}

contract Base2 {
    constructor() {
        test[-1] = 1;
        assert(test[-1] == 1);

        string memory s = "1";
        test2[s] = 1;
        assert(test2[s] == 1);
    }
    mapping(int => uint) test;
    mapping(string => uint) test2;
}

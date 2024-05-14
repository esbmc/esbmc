// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.5.0 <0.9.0;

contract A {
    int x;

    constructor() {
        x = 1;
    }
}

contract B is A {
    constructor() {
        x = 2;
    }
}

contract C is A, B {
    constructor() A() B() {}

    function test() public view {
        assert(x == 2);
    }
}

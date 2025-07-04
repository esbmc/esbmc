// SPDX-License-Identifier: GPL-3.0

pragma solidity >=0.8.2 ;

contract A {
    int x = 0;
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
    modifier func(int k, uint) {
        require(k == 2, "ok");
        _;
    }

    constructor() func(x, 2) A() B() {
        assert(1 == 0);
    }

    function test() public pure {}
}

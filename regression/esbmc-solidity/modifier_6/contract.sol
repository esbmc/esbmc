// SPDX-License-Identifier: GPL-3.0

pragma solidity >=0.8.2 <0.9.0;

contract A {
    int public x = 0;
    int public y = 5;

    constructor() {
        x = 1;
        y = 10;
    }

    modifier checkX(int expected) {
        require(x == expected, "x mismatch in A");
        _;
    }
}

contract B is A {
    int public z = 3;

    constructor() {
        x = 2;
        z = 4;
    }

    modifier checkZ(int expected) {
        require(z == expected, "z mismatch in B");
        _;
    }
}

contract C is A, B {
    int public k = 7;

    modifier outerModifier(int m, int n) {
        require(m + n == 12, "outer modifier failed");
        _;
    }

    modifier innerModifier() {
        require(y == 10, "inner modifier failed");
        _;
    }

    modifier deepestModifier(int p) {
        require(p == 7, "deepest modifier failed");
        _;
    }

    constructor()
        outerModifier(x, 10)
        innerModifier()
        deepestModifier(k)
        A()
        B()
    {
        assert(k == 7);
    }
}

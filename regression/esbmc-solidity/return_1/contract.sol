// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.5.0;

contract SolidityTest {
    uint8 x;
    uint8 y;

    constructor() {
        x = 0;
        y = 1;
    }

    function A() public returns (uint8) {
        return 11;
    }

    function B() public returns (uint8) {
        return y + A();
    }
}

contract Derived is SolidityTest {
    function getResult() public {
        SolidityTest xx = new SolidityTest();

        assert(xx.B() == 12);
    }
}

// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.5.0;

contract A {
    int x = 8;

    function setA() external {
        ++x;
        x++;
    }
}

contract B {
    uint internal mul;

    function setB() external {
        uint a = 2;
        uint b = 20;
        mul = a * b;
    }
}

contract C is A, B {
    function getStr() external {
        assert(x == 10);
    }

    function getMul() external {
        assert(mul == 40);
    }
}

contract caller {
    C contractC = new C();

    function testInheritance() public {
        contractC.setA();
        contractC.setB();
        contractC.getStr();
        contractC.getMul();
    }
}

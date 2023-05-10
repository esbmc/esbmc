// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.5.0;

// Methods overloading, contract_A.poly(-2) should call the poly(int8) while contract_A.poly(300) should call poly(uint16).
contract A {

    function poly(uint16 b) public pure returns (uint16) {
        return b;
    }

    function poly(int8 a) public pure returns (int8) {
        return a;
    }
}

contract C {
    function test_poly() public {
        A contract_A = new A();
        assert(contract_A.poly(-2) == -2);
        assert(contract_A.poly(300) == 300);
    }
}
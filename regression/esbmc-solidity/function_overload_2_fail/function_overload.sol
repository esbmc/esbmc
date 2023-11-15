// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.5.0;

// Methods overloading: contract_A.poly(-2) should call poly(int8) 
// while contract_A.poly(300) should call poly(uint16).
contract A {

    function func_1(uint16 b) public pure returns (uint16) {
        return b;
    }

    function func_1(int8 a) public pure returns (int8) {
        return a;
    }
}

contract C {
    function test_overload() public {
        A contract_A = new A();
        assert(contract_A.func_1(-2) == -2);
        assert(contract_A.func_1(300) < 300); // Fail because 300 is in range of uint16
    }
}
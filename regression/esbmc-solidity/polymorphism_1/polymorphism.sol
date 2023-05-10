// SPDX-License-Identifier: GPL-3.0

// Different contracts with the same function name. 
pragma solidity >=0.5.0;

contract A {

    function poly() public pure returns (uint8) {
        return 21;
    }
}

contract B {
    function poly() public pure returns (uint8) {
        return 42;
    }
}

contract C {
    function test_poly() public {
        A contract_A = new A();
        B contract_B = new B();
        assert(contract_A.poly() == 21);
        assert(contract_B.poly() == 42);
    }
}
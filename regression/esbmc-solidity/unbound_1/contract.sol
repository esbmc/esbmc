// SPDX-License-Identifier: GPL-3.0

// Different contracts with the same function name. 
pragma solidity >=0.5.0;

contract A {

    function func_1() public pure returns (uint8) {
        return 21;
    }
}

contract B {
    function func_1() public pure returns (uint8) {
        return 42;
    }
}

contract C {
    function test_overload() public {
        A contract_A = new A();
        B contract_B = new B();
        assert(contract_A.func_1() == 42); // FAIL because should return 21.
        assert(contract_B.func_1() == 21); // FAIL because should return 42.
    }
}
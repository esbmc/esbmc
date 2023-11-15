// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.5.0;

// Inheritance: Calls implicitly inherited parent's method.
contract A {
    function func_1() public pure returns (int8) {
        return 21;
    }
}

contract B is A {
}

contract call {
    function test_inheritance() public {
        A contract_a = new A();
        B contract_b = new B();
        assert(contract_a.func_1() == 21);
        assert(contract_b.func_1() == 21);
    }
}
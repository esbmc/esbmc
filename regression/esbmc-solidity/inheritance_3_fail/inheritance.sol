// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.5.0;

// Inheritance: Overrides grandparent's virtual method.
contract A {
    function func_1() public virtual returns (int8) {
        return 21;
    }
}

contract B is A {
}

contract C is B {
    function func_1() public override pure returns (int8) {
        return 42;
    }
}

contract call {
    function test_override() public {
        A contract_a = new A();
        B contract_b = new B();
        C contract_c = new C();
        assert(contract_a.func_1() == 21);
        assert(contract_b.func_1() == 21);
        assert(contract_c.func_1() == 21); // Fail because the contract_c.func_1() should return 42.
    }
}
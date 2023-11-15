// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.5.0;

// Inheritance: Overrides virtual from parent.
contract A {
    function func_1() public virtual returns (int8) {
        return 21;
    }
}

contract B is A {
    function func_1() public override pure returns (int8) {
        return 42;
    }
}

contract C {
    function test_override() public {
        A contract_A = new A();
        B contract_B = new B();
        assert(contract_A.func_1() == 42); // Fail because contract_A.func_1() should return 21
        assert(contract_B.func_1() == 42);
    }
}
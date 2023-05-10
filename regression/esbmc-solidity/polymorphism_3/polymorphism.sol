// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.5.0;

// Polymorphism: method overrides another virtual method with the same name from parent.
contract A {
    function poly() public virtual returns (int8) {
        return 21;
    }
}

contract B is A {
    function poly() public override pure returns (int8) {
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
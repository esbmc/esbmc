// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.5.0;

// Polymorphism: overrides grandparent's virtual method.
contract A {
    function poly() public virtual returns (int8) {
        return 21;
    }
}

contract B is A {
}

contract C is B {
    function poly() public override pure returns (int8) {
        return 42;
    }
}

contract call {
    function test_poly() public {
        C contract_c = new C();
        assert(contract_c.poly() == 42);
    }
}
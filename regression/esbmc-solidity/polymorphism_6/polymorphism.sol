// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.5.0;

// Polymorphism: Calling parent's method.
contract A {
    function poly() public pure returns (int8) {
        return 21;
    }
}

contract B is A {
}

contract call {
    function test_poly() public {
        B contract_b = new B();
        assert(contract_b.poly() == 21);
    }
}
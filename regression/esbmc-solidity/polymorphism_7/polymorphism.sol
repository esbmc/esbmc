// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.5.0;

// Polymorphism: Access state variable using inherited method after parent update.
contract A {

    uint8 a = 0;

    function get_num() public view returns (uint8) { return a; }

    function add_num() public virtual { a += 10; }
}

contract B is A {

    function add_num() public override { a += 20; }
}

contract call {
    function test_poly() public {        
        A contract_a = new A();
        B contract_b = new B();

        contract_a.add_num();
        // Contract B has a local cloned variable "a"
        // b.get_num() will return the cloned variable "a"
        // A.add_num() should not affect b.get_num()
        assert(contract_b.get_num() == 0); 
    }
}
// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.4.21;

// Inheritance: Access state variable using implicitly inherited method after state variables are updated.
contract A {

    uint8 a = 0;

    function get_num() public view returns (uint8) { return a; }

    function add_num() public virtual { a += 10; }
}

contract B is A {

	// KNOWNBUG a+20 could be overflow
    function add_num() public override { a += 20; }
}

contract Base {
    function test() public {        
        A contract_a = new A();
        B contract_b = new B();

        contract_a.add_num();
        // Contract B has a local cloned variable "a"
        // contract_b.get_num() should return the cloned variable "a"
        // contract_a.add_num() should not affect contract_b.get_num()
        assert(contract_a.get_num() == 10); 
        assert(contract_b.get_num() == 0); 

        // Similarly, contract_b.add_num() shoud not affect contract_a.get_num();
        contract_b.add_num();
        assert(contract_a.get_num() == 10);
        assert(contract_b.get_num() > 20);
    }
}

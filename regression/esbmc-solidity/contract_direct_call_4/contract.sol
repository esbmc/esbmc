// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract A {
    uint public balance;

    function deposit(uint amount) public virtual {
        require(amount < type(uint).max);
        require(balance < type(uint).max - amount);
        balance += amount ;
        assert(balance >= amount);
    }
}

contract B is A {
    function deposit(uint amount) public override {
        A.deposit(amount); 
    }
}
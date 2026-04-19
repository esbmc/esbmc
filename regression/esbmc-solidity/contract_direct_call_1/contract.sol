// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract A {
    uint public value;

    // Non-pure, non-view function that modifies state
    function setValue(uint _v) public virtual {
        value = _v;
        assert(1 == value);
    }
}

contract B is A {
    function setValue(uint _v) public override {
        A.setValue(_v); 
    }
}
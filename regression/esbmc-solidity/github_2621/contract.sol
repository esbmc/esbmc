// SPDX-License-Identifier: GPL-3.0
pragma solidity ^0.8.0;

contract B {
    uint key;
    uint value;

    function set(uint _key, uint _value) public {
        key = _key;
        value = _value;
        assert(key > value);
    }
}

contract C {
    B b= new B();

    function f() public {
        b.set(2, 3);
    }
}
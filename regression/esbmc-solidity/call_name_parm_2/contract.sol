// SPDX-License-Identifier: GPL-3.0
pragma solidity ^0.8.0;

contract C {

    uint key;
    uint value;

    function set(uint _key, uint _value) public {
        key = _key;
        value = _value;
    }

    function f() public {
        set({_value: 2, _key: 3});
        assert (key > value); 
    }
}
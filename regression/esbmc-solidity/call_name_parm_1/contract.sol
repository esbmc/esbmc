// SPDX-License-Identifier: GPL-3.0
pragma solidity ^0.8.0;

contract C {

    uint key;
    uint value;
    uint test;

    function set(uint _key, uint _value, uint _test) public {
        key = _key;
        value = _value;
        test = _test;
    }

    function f() public {
        set({_test: 4, _value: 2, _key: 3});
        assert (key > value); 
        assert (test == 3);
    }
}
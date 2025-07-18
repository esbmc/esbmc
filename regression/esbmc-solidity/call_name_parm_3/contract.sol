// SPDX-License-Identifier: GPL-3.0
pragma solidity ^0.8.0;


contract C {
    mapping(uint => uint) data;

    function set(uint key, uint value) public {
        data[key] = value;
    }
    
    function f() public {
        set({value: 2, key: 3});
        assert (data[3] == 3);
    }
}
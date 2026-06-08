// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.8.0;

contract NamedArgsFail {
    mapping(uint => uint) data;

    function set(uint key, uint value) public {
        data[key] = value;
    }

    function test() public {
        // named arguments: key=7, value=42
        set({value: 42, key: 7});
        // wrong: data[7] is 42, not 99
        assert(data[7] == 99);
    }
}

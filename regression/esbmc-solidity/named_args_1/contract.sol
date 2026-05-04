// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.8.0;

contract NamedArgs {
    mapping(uint => uint) data;

    function set(uint key, uint value) public {
        data[key] = value;
    }

    function test() public {
        // named arguments in reverse order
        set({value: 42, key: 7});
        assert(data[7] == 42);
    }
}

// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.8.0 <0.9.0;

contract DynPushFail {
    uint[] public items;

    function test() public {
        items.push(100);
        items.push(200);
        // wrong value: should be 100, not 999
        assert(items[0] == 999);
    }
}

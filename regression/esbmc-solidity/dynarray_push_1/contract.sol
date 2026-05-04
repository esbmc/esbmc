// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.8.0 <0.9.0;

contract DynPush {
    uint[] public items;

    function test() public {
        items.push(100);
        items.push(200);
        items.push(300);
        assert(items[0] == 100);
        assert(items[1] == 200);
        assert(items[2] == 300);
        assert(items.length == 3);

        items.pop();
        assert(items.length == 2);
    }
}

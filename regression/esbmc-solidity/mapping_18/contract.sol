// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.8.0;

contract MappingInStruct {
    struct Record {
        mapping(uint => uint) balances;
        uint256 count;
    }

    Record public rec;

    function test() public {
        rec.balances[1] = 10;
        rec.balances[2] = 20;
        rec.count = 2;

        assert(rec.balances[1] == 10);
        assert(rec.balances[2] == 20);
        assert(rec.count == 2);
    }
}

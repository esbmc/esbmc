// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.8.0;

contract NestedMappingInStruct {
    struct Ledger {
        mapping(uint => mapping(uint => uint)) records;
        uint256 total;
    }

    Ledger public ledger;

    function test() public {
        ledger.records[1][2] = 100;
        ledger.total = 1;

        assert(ledger.records[1][2] == 100);
        assert(ledger.total == 1);
    }
}

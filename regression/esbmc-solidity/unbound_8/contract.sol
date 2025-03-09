// SPDX-License-Identifier: GPL-3.0
pragma solidity ^0.8.10;

contract IntegerOverflowMultiTxMultiFuncFeasible {
    uint8 public initialized = 0;
    uint8 public count = 1;

    function init() public {
        initialized = 1;
    }

    function run(uint8 input) public {
        this.init();
        assert(this.initialized() != this.count());
    }
}


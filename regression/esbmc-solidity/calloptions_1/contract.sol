// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.8.0;

// Test: FunctionCallOptions with reordered {gas, value}
// Verifies that the frontend correctly extracts the "value" option
// by name rather than by position in the options array.

contract Receiver {
    function deposit() public payable {}
}

contract Caller {
    Receiver target;

    constructor(address _target) {
        target = Receiver(_target);
    }

    // {value: X, gas: Y} — value first (standard order)
    function sendValueFirst() public payable {
        uint before = address(target).balance;
        target.deposit{value: msg.value, gas: 100000}();
        uint after_ = address(target).balance;
        assert(after_ == before + msg.value);
    }

    // {gas: Y, value: X} — gas first (reordered)
    function sendGasFirst() public payable {
        uint before = address(target).balance;
        target.deposit{gas: 100000, value: msg.value}();
        uint after_ = address(target).balance;
        assert(after_ == before + msg.value);
    }
}

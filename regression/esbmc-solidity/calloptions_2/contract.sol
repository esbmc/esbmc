// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.8.0;

// Test: FunctionCallOptions with reordered {gas, value} in unbound mode
// In unbound mode the external call result is over-approximated as nondet,
// so the balance assertion is expected to fail.

contract Receiver {
    function deposit() public payable {}
}

contract Caller {
    Receiver target;

    constructor(address _target) {
        target = Receiver(_target);
    }

    // {gas: Y, value: X} — reordered options
    function sendGasFirst() public payable {
        uint before = address(target).balance;
        target.deposit{gas: 100000, value: msg.value}();
        uint after_ = address(target).balance;
        assert(after_ == before + msg.value);
    }
}

// SPDX-License-Identifier: GPL-3.0
pragma solidity ^0.8.0;

interface ICallback { function cb() external; }

contract Victim {
    bool public entered;
    bool public reentered;
    function run() external {
        if (entered) { reentered = true; }   // true only on re-entry
        entered = true;
        ICallback(msg.sender).cb();           // address-cast callback
        entered = false;
        assert(!reentered);                   // violated iff run() re-enters
    }
}

contract Attacker is ICallback {
    Victim v;
    bool private done;
    constructor(address _v) { v = Victim(_v); }
    function cb() external override { if (!done) { done = true; v.run(); } }
    function Exploit() external { v.run(); }
}

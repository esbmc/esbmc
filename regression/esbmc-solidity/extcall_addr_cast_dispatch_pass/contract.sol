// SPDX-License-Identifier: GPL-3.0
pragma solidity ^0.8.0;

interface ICallback { function onCall(uint256 v) external; }

// Companion to cast_contract_member_call_bound, but the cast operand is
// `msg.sender` — an address-valued expression with NO referencedDeclaration,
// the case the dispatch fix adds. The callee runs (writes only its own
// state) and does not re-enter, so Victim's own state is untouched and the
// property holds: the fix lowers the address-cast call without a false
// positive.
contract Victim {
    uint256 public marker = 111;
    function run(uint256 v) external {
        ICallback(msg.sender).onCall(v);  // address-cast on msg.sender (the fix)
        assert(marker == 111);            // Victim state untouched by external call
    }
}

contract Caller is ICallback {
    Victim victim;
    uint256 public last;
    constructor(address _v) { victim = Victim(_v); }
    function onCall(uint256 v) external override { last = v; }  // no re-entry
    function go(uint256 v) external { victim.run(v); }
}

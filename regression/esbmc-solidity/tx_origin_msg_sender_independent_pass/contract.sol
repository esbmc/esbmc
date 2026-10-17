// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

// Counterpart to `tx_origin_msg_sender_independent_fail`.  Users who
// want to restrict the search to direct EOA -> contract calls (real
// EVM invariant: `tx.origin == msg.sender` at the top of every
// transaction) can opt in with an explicit `require`.  With that
// constraint the equality assertion trivially holds, so this test
// must PASS.  Together the two tests pin down the new model:
// independent by default, equal on explicit user request.
contract Equal {
    uint256 public x;
    function f(uint256 v) external {
        require(tx.origin == msg.sender);
        assert(msg.sender == tx.origin);
        x = v;
    }
}

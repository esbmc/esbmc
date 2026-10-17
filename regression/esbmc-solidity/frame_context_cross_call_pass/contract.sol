// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

// B2 — Frame-context consistency across an external call.  When
// contract A calls a payable method on contract B (or transfers
// ETH via .transfer() / .send() / address(B).call{value:v}()),
// the EVM swaps `msg.sender` and `msg.value` to B's frame for the
// duration of the call and restores them when the call returns.
// ESBMC models this with a save/set/restore wrap in
// `solidity_convert_call.cpp` (see `model_transaction` and the
// .call/.delegatecall/.staticcall lowering paths).  This test
// asserts the invariant end-to-end: A reads msg.sender, calls B,
// then re-reads msg.sender — the two reads must agree because the
// wrap restores the global after B's frame exits.
contract Sink {
    uint public received;
    receive() external payable {
        received += msg.value;
    }
}

contract A {
    Sink immutable sink;

    constructor() payable {
        sink = new Sink();
    }

    function pay(uint amt) public payable {
        if (address(this).balance < amt) return;
        if (amt == 0) return;
        address before = msg.sender;
        uint vbefore = msg.value;
        payable(address(sink)).transfer(amt);
        address after_ = msg.sender;
        uint vafter = msg.value;
        // msg.sender / msg.value belong to A's frame both before
        // and after the transfer — they must be unchanged by Sink's
        // receive() executing in between.
        assert(before == after_);
        assert(vbefore == vafter);
    }
}

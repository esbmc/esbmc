// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

// Verifies the EOA-recipient fallback in solidity_convert_call.cpp's
// transfer model.  Before the fix, addr.transfer(v) to an unknown
// (non-tracked) address fell through `return false` with no balance
// change — sender's balance accounting was wrong on every payment to
// an EOA.  After the fix, the sender's balance MUST decrease by `v`.
contract Bal {
    constructor() payable {}

    function payAndCheck(address payable to, uint amt) public {
        // Skip the matched-recipient self-transfer case (`to ==
        // address(this)`), which deducts AND credits the same balance
        // and would mask the EOA-deduct property we want to check.
        if (to == address(this)) return;
        // Assume enough balance and a meaningful payment.
        if (address(this).balance < amt || amt == 0) return;
        uint before = address(this).balance;
        to.transfer(amt);
        uint after_ = address(this).balance;
        // Sender's balance must drop by exactly `amt`, regardless of
        // whether `to` is a tracked contract instance or an EOA.
        assert(after_ == before - amt);
    }
}

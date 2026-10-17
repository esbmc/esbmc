// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

// Ledger #16: a partial-write leak via SSA merge would let the post-state
// "x == 1 and balance NOT debited" be observable from a sibling dispatcher
// iteration. Real EVM forbids this (success debits both); ESBMC's
// assume(false) prunes the failure branch so the leak is unreachable too.
contract Vault {
    uint public x;
    uint public bal_at_pay;
    function pay(address payable to, uint amt) public {
        bal_at_pay = address(this).balance;
        x = 1;
        to.transfer(amt);
    }
    receive() external payable {}
}

contract H {
    Vault a;
    Vault b;
    constructor() {
        a = new Vault();
        b = new Vault();
    }
    function trigger(uint amt) public {
        a.pay(payable(address(b)), amt);
    }
    function check() public view {
        // If x == 1, then by post-success semantics a's balance must have
        // been at least bal_at_pay (the value seen when pay started). A
        // partial-write leak would let x == 1 coexist with a's balance
        // exceeding bal_at_pay (which would mean the debit didn't happen).
        if (a.x() == 1) {
            assert(address(a).balance <= a.bal_at_pay());
        }
    }
}

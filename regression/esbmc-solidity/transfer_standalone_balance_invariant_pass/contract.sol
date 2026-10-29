// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

// Ledger #16: balance side. Two tracked Vaults, all transfers between them.
// Total balance preserved across iters. A partial-write leak in transfer's
// failure path would let the debit happen without the credit (or vice versa).
contract Vault {
    function pay(address payable to, uint amt) public {
        to.transfer(amt);
    }
    receive() external payable {}
}

contract H {
    Vault a;
    Vault b;
    uint initial_total;
    constructor() {
        a = new Vault();
        b = new Vault();
        initial_total = address(a).balance + address(b).balance;
    }
    function bridge_a_to_b(uint amt) public {
        a.pay(payable(address(b)), amt);
    }
    function bridge_b_to_a(uint amt) public {
        b.pay(payable(address(a)), amt);
    }
    function check_total_invariant() public view {
        // Total preserved: every successful transfer debits one and credits
        // the other; failed transfers are pruned (no commit at all).
        assert(address(a).balance + address(b).balance == initial_total);
    }
}

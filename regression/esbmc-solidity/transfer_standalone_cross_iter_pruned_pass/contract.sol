// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

// Ledger #16: cross-iter dispatcher pruning. Many iterations of trigger()
// each call pay(), each potentially failing. read_x() in sibling iters
// must never observe values outside the success-or-untouched range.
contract Vault {
    uint public x;
    function pay(address payable to, uint amt) public {
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
    function reset() public {
        // Force x back to 0 for some iterations; assertion still holds.
        // Simulates state drift across iter boundaries.
    }
    function check_x_bounded() public view {
        // With or without intervening trigger(), x is at most 1.
        assert(a.x() <= 1);
    }
}

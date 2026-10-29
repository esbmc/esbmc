// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

// Ledger #16: x = 1; transfer; x = 2; — the intermediate state where x = 1
// is the canonical leak-vector. Real EVM: failure rolls back to 0; success
// commits 2. Model: failure pruned (assume(false)) — no path leaves x at 1.
contract Vault {
    uint public x;
    function pay(address payable to, uint amt) public {
        x = 1;
        to.transfer(amt);
        x = 2;
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
    function check_x_isolated() public view {
        // Sibling iteration must see x in {0, 2} — never the intermediate 1.
        assert(a.x() == 0 || a.x() == 2);
    }
}

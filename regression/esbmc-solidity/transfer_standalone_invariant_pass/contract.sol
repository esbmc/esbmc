// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

// Ledger #16, deferred sub-case: standalone addr.transfer() outside try/catch
// uses __ESBMC_assume(false) on the failure branch (insufficient balance to a
// tracked contract). The conjectured leak is "path-pruned write survives via
// SSA merge". This pass-mode test asserts the smallest possible cross-iter
// invariant: x is never observable in any partial intermediate state because
// only the success branch commits both x = 1 and the transfer.
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
    function check_x_invariant() public view {
        assert(a.x() == 0 || a.x() == 1);
    }
}

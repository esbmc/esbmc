// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

// Negative counterpart of eoa_balance_credit_pass.  After we transfer
// `amt > 0` to the recipient, asserting balance is unchanged must
// fail — the EOA balance map credits the recipient.
contract Bal {
    constructor() payable {}

    function check(address payable to, uint amt) public {
        if (to == address(this)) return;
        if (address(this).balance < amt || amt == 0) return;
        if (to.balance != 0) return;
        to.transfer(amt);
        // Wrong: balance grew by `amt`, so == 0 is false for amt > 0.
        assert(to.balance == 0);
    }
}

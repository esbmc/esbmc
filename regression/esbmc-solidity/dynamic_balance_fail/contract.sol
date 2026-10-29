// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

// Companion failing case to dynamic_balance_pass: same source structure
// but the assertion is intentionally wrong (claims the balance does NOT
// move). With the cpp_new lowering fix, address(.).balance correctly
// reflects the 30-wei withdraw, so `pre == post` (no movement) is
// falsifiable.

contract Vault {
    constructor() payable {}
    function withdraw(address payable to, uint256 amt) external {
        to.transfer(amt);
    }
}

contract Probe {
    function check() external payable {
        Vault v = new Vault{value: 100}();
        address a = address(v);
        uint pre  = a.balance;
        v.withdraw(payable(address(0x1234)), 30);
        uint post = a.balance;
        assert(pre == post);   // wrong — withdraw should move 30 wei out
    }
}

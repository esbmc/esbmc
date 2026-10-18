// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

// Companion FAIL case to dynamic_balance_state_var_pass: same source
// shape but the assertion claims the balance is unchanged after the
// withdraw. With the fix, address(v).balance correctly tracks the
// 30-wei decrement, so `pre == post` is falsifiable.

contract Vault {
    constructor() payable {}
    function withdraw(address payable to, uint256 amt) external {
        to.transfer(amt);
    }
}

contract Probe {
    Vault v;
    constructor() payable { v = new Vault{value: 100}(); }
    function check() external {
        uint pre  = address(v).balance;
        v.withdraw(payable(address(0x1234)), 30);
        uint post = address(v).balance;
        assert(pre == post);   // wrong — 30 wei should be moved out
    }
}

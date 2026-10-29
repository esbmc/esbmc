// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;
contract NonPay {
    uint256 public hit;
    function f(uint256 x) external {   // NOT payable
        if (x >= 100) { hit = 1; } else { hit = 2; }
    }
}

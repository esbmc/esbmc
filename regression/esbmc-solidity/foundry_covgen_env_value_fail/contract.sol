// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;
contract PayGate {
    uint256 public hit;
    function pay() external payable {
        if (msg.value >= 1000000000000000000) {  // 1 ether
            hit = 1;
        } else {
            hit = 2;
        }
    }
}

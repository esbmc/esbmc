// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;
contract RecvC {
    uint256 public hit;
    receive() external payable {
        if (msg.value >= 1000000000000000000) { hit = 1; } else { hit = 2; }
    }
    function ping() external { hit = 3; }
}

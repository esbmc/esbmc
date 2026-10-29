// SPDX-License-Identifier: MIT
pragma solidity ^0.8.30;

// Foundry coverage-test generation: ESBMC reconstructs a compilable *.t.sol
// that replays the concrete inputs covering each branch of `set`.
contract Cov {
    uint256 public x;
    function set(uint256 v, bool flag) public {
        if (flag && v > 10) {
            x = v;
        } else {
            x = 0;
        }
    }
}

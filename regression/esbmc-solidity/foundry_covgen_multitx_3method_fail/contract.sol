// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;
contract Multi3 {
    uint256 public flag;
    uint256 public r;
    function noise1() external { r = 100; }
    function noise2() external { r = 200; }
    function setState() external { flag = 7; }
    function gate() external {
        if (flag == 7) r = 1;   // reachable only after setState()
        else r = 2;
    }
}

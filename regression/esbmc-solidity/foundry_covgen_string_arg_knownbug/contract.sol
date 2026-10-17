// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;
contract StrArg {
    uint256 public v;
    function setName(string memory n) public { if (bytes(n).length > 3) v = 1; else v = 2; }
}

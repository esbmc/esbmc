// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;
contract BytesArg {
    uint256 public v;
    function setData(bytes memory d) public { if (d.length > 3) v = 3; else v = 4; }
}

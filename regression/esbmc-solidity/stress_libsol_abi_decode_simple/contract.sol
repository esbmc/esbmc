// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.8.24;
contract C {
    function f(bytes memory data) public pure returns (uint256, bytes memory) {
        return abi.decode(data, (uint256, bytes));
    }
}

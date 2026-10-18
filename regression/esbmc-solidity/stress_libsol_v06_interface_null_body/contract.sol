// SPDX-License-Identifier: MIT
pragma solidity ^0.6.0;

interface IThing {
    function get() external view returns (uint256);
}

contract T is IThing {
    function get() external view override returns (uint256) {
        return 0;
    }

    function check(uint256 v) public pure {
        assert(v == v);
    }
}

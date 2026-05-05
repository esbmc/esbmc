// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

interface ISimple {
    function getValue() external view returns (uint256);
}

contract Test {
    function check() public pure {
        bytes4 id = type(ISimple).interfaceId;
        // interfaceId is bytes4; with nondet, assert reflexive equality
        assert(id == id);
    }
}

// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

// Ledger #15: type(I).interfaceId is currently a fresh nondet per call.
// Two reads of the same interface's id can return different values,
// breaking real-EVM semantics where interfaceId is a compile-time
// constant per interface.
interface ISimple {
    function getValue() external view returns (uint256);
}

contract Test {
    function check() public pure {
        // Two SEPARATE reads of type(ISimple).interfaceId.
        bytes4 id1 = type(ISimple).interfaceId;
        bytes4 id2 = type(ISimple).interfaceId;
        assert(id1 == id2);
    }
}

// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

interface ISimple {
    function getValue() external view returns (uint256);
}

contract Test {
    function check() public pure {
        bytes4 id = type(ISimple).interfaceId;
        uint32 idVal = uint32(id);
        // nondet interfaceId cast to uint32: asserting == 0 should fail
        assert(idVal == 0);
    }
}

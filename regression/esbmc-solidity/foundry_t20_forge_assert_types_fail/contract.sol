// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;
abstract contract Test {
    function assertEq(bool a, bool b) internal pure { require(a == b); }
    function assertEq(address a, address b) internal pure { require(a == b); }
    function assertEq(bytes32 a, bytes32 b) internal pure { require(a == b); }
    function assertEq(string memory a, string memory b) internal pure {
        require(keccak256(bytes(a)) == keccak256(bytes(b)));
    }
}
contract CT is Test {
    // bytes32 mismatch → FAILED (type-specific equality actually checked)
    function test_b32_bad() public pure {
        assertEq(bytes32(uint256(7)), bytes32(uint256(8)));
    }
}

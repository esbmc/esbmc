// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;
// forge-std assertEq is overloaded per operand type. ESBMC intercepts assertEq
// BY NAME and lowers to assert(a == b); its generic equality path already
// covers bool / address / bytes32 / string (content-wise) — verified against
// real forge 1.7.1. The stub bodies below exist only so solc emits a valid AST.
abstract contract Test {
    function assertEq(bool a, bool b) internal pure { require(a == b); }
    function assertEq(address a, address b) internal pure { require(a == b); }
    function assertEq(bytes32 a, bytes32 b) internal pure { require(a == b); }
    function assertEq(string memory a, string memory b) internal pure {
        require(keccak256(bytes(a)) == keccak256(bytes(b)));
    }
}
contract CT is Test {
    function test_types_ok() public pure {
        assertEq(true, true);
        assertEq(address(0x1), address(0x1));
        assertEq(bytes32(uint256(7)), bytes32(uint256(7)));
        assertEq(string("hi"), string("hi"));
    }
}

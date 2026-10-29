// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;
// forge-std assertEq overloads for reference types. ESBMC intercepts by name:
// bytes -> precise content compare (bytes_dynamic_equal); arrays -> conservative
// prune (no false WRONG). Stub bodies exist only so solc emits a valid AST.
abstract contract Test {
    function assertEq(bytes memory a, bytes memory b) internal pure {
        require(keccak256(a) == keccak256(b));
    }
    function assertEq(uint256[] memory a, uint256[] memory b) internal pure {
        require(a.length == b.length);
        for (uint256 i = 0; i < a.length; i++) require(a[i] == b[i]);
    }
}
contract CT is Test {
    // content-equal dynamic bytes -> SUCCESSFUL (precise, matches forge PASS)
    function test_bytes_ok() public pure {
        assertEq(bytes(hex"0102"), bytes(hex"0102"));
    }
}

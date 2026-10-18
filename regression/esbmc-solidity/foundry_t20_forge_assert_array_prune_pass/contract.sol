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
    // array assertEq is conservatively pruned. Before the fix the generic
    // reference-equality path reported this content-EQUAL case as FAILED
    // (a false WRONG); pruning yields SUCCESSFUL, never a false WRONG.
    function test_arr_pruned() public pure {
        uint256[] memory a = new uint256[](2); a[0] = 1; a[1] = 2;
        uint256[] memory b = new uint256[](2); b[0] = 1; b[1] = 2;
        assertEq(a, b);
    }
}

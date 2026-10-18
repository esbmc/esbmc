// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;
// Minimal forge-std stub so solc produces a valid AST. ESBMC intercepts
// assertApproxEqAbs BY NAME and lowers it to assert(|a-b| <= maxDelta); the
// require-based stub body below is never used at verification time.
abstract contract Test {
    function assertApproxEqAbs(uint256 a, uint256 b, uint256 d) internal pure {
        require((a >= b ? a - b : b - a) <= d, "approx");
    }
}
contract Counter { uint256 public x; function set(uint256 v) public { x = v; } }
contract CT is Test {
    // |100 - 103| = 3 <= 5  → SUCCESSFUL
    function test_within() public {
        Counter c = new Counter(); c.set(103);
        assertApproxEqAbs(100, c.x(), 5);
    }
    // boundary |100 - 103| = 3 <= 3 → SUCCESSFUL
    function test_boundary() public {
        Counter c = new Counter(); c.set(103);
        assertApproxEqAbs(c.x(), 100, 3);
    }
}

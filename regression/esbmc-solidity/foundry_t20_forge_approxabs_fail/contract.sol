// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;
abstract contract Test {
    function assertApproxEqAbs(uint256 a, uint256 b, uint256 d) internal pure {
        require((a >= b ? a - b : b - a) <= d, "approx");
    }
}
contract Counter { uint256 public x; function set(uint256 v) public { x = v; } }
contract CT is Test {
    // |100 - 103| = 3 > 2 → FAILED (real approx-tolerance bug detection)
    function test_exceeds() public {
        Counter c = new Counter(); c.set(103);
        assertApproxEqAbs(100, c.x(), 2);
    }
}

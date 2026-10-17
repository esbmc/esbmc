// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;
abstract contract Test {
    function assertEq(uint256 a, uint256 b) internal pure { require(a == b, "eq"); }
    function assertTrue(bool c) internal pure { require(c, "true"); }
    function assertGt(uint256 a, uint256 b) internal pure { require(a > b, "gt"); }
}
contract Counter { uint256 public x; function inc() public { x += 1; } }
contract CT is Test {
    function test_eq_ok() public {
        Counter c = new Counter(); c.inc();
        assertEq(c.x(), 1);          // correct → SUCCESSFUL
    }
    function test_eq_wrong() public {
        Counter c = new Counter(); c.inc();
        assertEq(c.x(), 2);          // wrong expectation → FAILED (real Foundry test detection!)
    }
    function test_true_wrong() public {
        Counter c = new Counter();
        assertTrue(c.x() == 5);      // x==0 → FAILED
    }
    function test_gt_ok() public {
        Counter c = new Counter(); c.inc();
        assertGt(c.x(), 0);          // 1>0 → SUCCESSFUL
    }
}

// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;
abstract contract Test {
  function assertEqDecimal(uint256 a, uint256 b, uint256) internal pure { require(a==b); }
  function assertEqUint(uint256 a, uint256 b) internal pure { require(a==b); }
  function assertGtDecimal(uint256 a, uint256 b, uint256) internal pure { require(a>b); }
}
contract D is Test {
  function test_eqDec_ok() public { assertEqDecimal(100, 100, 18); }
  function test_eqDec_wrong() public { assertEqDecimal(100, 101, 18); }
  function test_eqUint_ok() public { assertEqUint(5, 5); }
  function test_eqUint_wrong() public { assertEqUint(5, 6); }
  function test_gtDec_ok() public { assertGtDecimal(10, 3, 18); }
  function test_gtDec_wrong() public { assertGtDecimal(3, 10, 18); }
}

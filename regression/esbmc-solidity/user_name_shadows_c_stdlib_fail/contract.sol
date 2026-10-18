// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

// Dual of user_name_shadows_c_stdlib_pass: the same resolution must
// correctly propagate the user-defined div's uint256 semantics so a
// violating assertion is refuted.  The post-condition v/1 == v+1 is
// false; if resolution silently bound to C's div_t-returning div, the
// assertion would crash at the type-check instead of the assert.
library SafeMath {
  function div(uint256 a, uint256 b) internal pure returns (uint256) {
    return a / b;
  }
}

contract C {
  using SafeMath for uint256;

  function calc(uint256 v) public pure returns (uint256) {
    uint256 r = v.div(1);
    assert(r == v + 1);
    return r;
  }
}

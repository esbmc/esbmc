// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

// Regression: a Solidity library function whose name collides with a C
// stdlib.h symbol (here, `div` which returns div_t in C) must still
// resolve to the Solidity decl, not the C one.  Before the fix,
// get_sol_builtin_ref bound `div(...)` to `c:@F@div` from stdlib.h,
// yielding a struct return type and tripping
// "got struct, expected unsignedbv" at any caller expecting uint256.
library SafeMath {
  function div(uint256 a, uint256 b) internal pure returns (uint256) {
    return a / b;
  }
}

contract C {
  using SafeMath for uint256;

  function calc(uint256 v) public pure returns (uint256) {
    // Exercise the path where `div` is called as a user-defined name
    // in a position expecting a uint256; pre-fix this typechecks a
    // struct `div_t` against unsignedbv and aborts conversion.
    uint256 r = v.div(1);
    assert(r == v);
    return r;
  }
}

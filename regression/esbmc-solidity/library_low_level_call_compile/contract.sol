// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

// A low-level `.call(data)` inside a LIBRARY body (SafeERC20-style
// payload) used to crash ESBMC with
//     Function `sol:@C@<Lib>@F@$call#0' type mismatch: expected code
//     ERROR: failed to find function `sol:@C@<Lib>@F@$call#0'
// because `populate_low_level_functions` is only run for contracts,
// not libraries, so the per-contract `$call#0` helper was never
// registered when the lowering site looked it up.  Now the low-level
// lowering detects library scope and emits just the nondet return
// tuple (no dispatch ladder, which wouldn't apply to libraries
// anyway).
//
// This test only checks that compilation + symex reach a verdict —
// the body of the `.call` is over-approximated as nondet.

library Libby {
    function callOptionalReturn(address token, bytes memory data) internal {
        (bool success, bytes memory returndata) = token.call(data);
        require(success);
        returndata;
    }
}

contract C {
    function test(address token, bytes memory data) public {
        Libby.callOptionalReturn(token, data);
    }
}

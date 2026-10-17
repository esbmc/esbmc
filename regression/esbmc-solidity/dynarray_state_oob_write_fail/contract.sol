// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.5.0;

// Writing past the length of a dynamic-array state variable must be caught as
// an array-bounds violation under --bounds-check. Same get_index_access_expr
// lowering serves the assignment LHS, so the explicit bounds claim fires for a
// write too (regression pin for the Solidity dyn-array OOB fix, write path).
contract Base {
    uint[] a;
    function run() public {
        a.push(7);
        a[5] = 9;   // OUT OF BOUNDS: length is 1
    }
}

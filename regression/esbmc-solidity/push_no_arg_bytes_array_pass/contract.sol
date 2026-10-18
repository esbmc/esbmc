// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.8.0;

// P1 regression-lock: `bytes[] arr; arr.push();` currently CRASHES.
// Same root as the struct-element form — `bytes[]` element type is
// `BytesDynamic`, stored as `symbol_typet`; gen_zero returns nil.
//
// First surfaced 2026-04-30 while landing the delete-correctness work
// (memory `project_push_pop_pre_existing_bug.md`).
//
// KNOWNBUG until S1+S2 land.
contract C {
    bytes[] arr;

    function f() public {
        require(arr.length == 0);
        arr.push();
        assert(arr.length == 1);
        assert(arr[0].length == 0);
    }
}

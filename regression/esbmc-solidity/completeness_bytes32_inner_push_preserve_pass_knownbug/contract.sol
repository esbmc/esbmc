// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

// Completeness pin (KNOWNBUG): bytes32 inner-array push preservation.
//
// Asserts that after 3 inner-array pushes onto a bytes32[][]
// state-var nested dynamic array, OLD slot reads still match the
// originally-pushed values.
//
// CURRENT STATE (2026-05-10): symex CRASHES with
// `Fatal glibc error: malloc.c:2599 (sysmalloc): assertion failed`
// at __memcpy_impl iteration 40+ during incremental-bmc. The per-
// byte memcpy preservation chain in _ESBMC_array_push (via realloc
// + __builtin_memcpy) interacts badly with glibc's malloc internals
// at this scale — preservation isn't just slow, it's broken before
// the SAT solver even gets the formula.
//
// This pin documents the bug. Stays KNOWNBUG until either:
//   (a) the model's allocator/memcpy shape stops crashing, or
//   (b) per-byte preservation is replaced with a typed-element
//       loop avoiding realloc+memcpy at this scale (would change
//       the failure mode — review the new state before promoting).
contract C {
    bytes32[][] internal arr;

    function run() external {
        if (arr.length == 0)
            arr.push();
        uint256 base = arr[0].length;
        arr[0].push(bytes32(uint256(0x01)));
        arr[0].push(bytes32(uint256(0x02)));
        arr[0].push(bytes32(uint256(0x03)));
        // OLD-slot reads — preservation must hold
        assert(arr[0][base] == bytes32(uint256(0x01)));
        assert(arr[0][base + 1] == bytes32(uint256(0x02)));
        assert(arr[0][base + 2] == bytes32(uint256(0x03)));
    }
}

// SPDX-License-Identifier: UNLICENSED
pragma solidity >=0.8.0 <0.9.0;

// Violation test for 2D fully-fixed `bytes32[N][M]`.
//
// Fixed 2026-04-24. The previous "silent drop" was in the library:
// `bytes_static_equal` used `memcmp(a->data, b->data, 32)`, but
// string.c's memcmp is a byte-stepping loop that k-induction (and
// BMC with --unwind < 32) bounds to k iterations — silently
// truncating the comparison and producing false-positive equality
// for any bytes beyond the k-th position. For this test, the write
// to `buf[1][2]` never happens, so `buf[1][2]` is all-zero while
// the RHS `bytes32(uint256(0xbeef))` has 0xbeef in the last two
// bytes. With memcmp truncated to k=3 iterations, only the leading
// 3 bytes (all 0x00) were compared — comparison returned 0 (equal)
// and the assertion vacuously passed.
//
// Fix: rewrite `bytes_static_equal` in
// src/c2goto/library/solidity/solidity_bytes.c with 32 length-gated
// byte comparisons in a straight-line `&&` chain. No loop, no
// unwind dependency. Matches the existing pattern already used by
// `bytes_static_to_uint` for the same reason.
//
// The inductive step at k=4 generates 4 meaningful VCCs now, but
// cvc5 still times out on the resulting ~2500-assignment SSA with
// 2D array_typet selects. Test therefore runs under
// `--incremental-bmc`, which reaches VERIFICATION FAILED at k=33
// in ~20s once the dispatch loop unrolls enough to enter `run()`.
contract MultiDimBytes32_2DFail {
    bytes32[3][2] internal buf;

    function run() external {
        buf[0][0] = bytes32(uint256(0x1111));
        // BUG: buf[1][2] never written.
        assert(buf[1][2] == bytes32(uint256(0xbeef)));
    }
}

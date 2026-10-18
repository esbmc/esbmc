// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

// Single-push case for `uint8[3][]` as a struct field. Pinned by
// the cast + writeback fix at solidity_convert_ref.cpp:1346+1457.
//
// At HEAD before the fix, this was VERIFICATION FAILED because:
//   1. The aux pointer-to-row was bit-reinterpreted as `(unsigned char [3])`
//      so memcpy copied the pointer's bits, not the row data.
//   2. The realloc'd return pointer was not written back to
//      `bucket.rows`, so subsequent reads went through a NULL/stale
//      pointer.
//
// With the fix, the assertion `bucket.rows[0][0] == 1` holds after a
// single `bucket.rows.push(r)` where `r[0] = 1`.
contract C {
    struct Bucket { uint8[3][] rows; }
    Bucket internal bucket;

    function pushRow(uint8 a) internal {
        uint8[3] memory r;
        r[0] = a;
        bucket.rows.push(r);
    }

    function run() external {
        pushRow(1);
        assert(bucket.rows[0][0] == 1);
    }
}

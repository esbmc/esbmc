// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

// Companion FAIL test for napp_struct_fixdyn_uint8_1push_pass.
// Pushes 1, then asserts the row holds 2 — the assertion is intended
// to fail (counter-example with row=[1,0,0]).
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
        assert(bucket.rows[0][0] == 2);
    }
}

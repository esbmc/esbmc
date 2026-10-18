// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

// Dual-fail: identical struct + uint8[3][] harness with flipped
// post-pop length invariant.
contract C {
    struct Bucket {
        uint8 lid;
        uint8[3][] rows;
    }

    Bucket internal bucket;

    function pushRow(uint8 a, uint8 b, uint8 c) internal {
        uint8[3] memory r;
        r[0] = a;
        r[1] = b;
        r[2] = c;
        bucket.rows.push(r);
    }

    function run() external {
        bucket.lid = 5;
        assert(bucket.lid == 5);
        assert(bucket.rows.length == 0);

        pushRow(1, 2, 3);
        pushRow(4, 5, 6);
        pushRow(7, 8, 9);
        assert(bucket.rows.length == 3);
        assert(bucket.rows[0][0] == 1);
        assert(bucket.rows[0][1] == 2);
        assert(bucket.rows[0][2] == 3);
        assert(bucket.rows[1][0] == 4);
        assert(bucket.rows[1][1] == 5);
        assert(bucket.rows[1][2] == 6);
        assert(bucket.rows[2][0] == 7);
        assert(bucket.rows[2][1] == 8);
        assert(bucket.rows[2][2] == 9);

        bucket.rows[1][2] = 60;
        assert(bucket.rows[1][2] == 60);
        assert(bucket.rows[1][0] == 4);
        assert(bucket.rows[1][1] == 5);

        bucket.rows.pop();
        // FLIPPED: actual length is 2 after pop, not 3
        assert(bucket.rows.length == 3);
        assert(bucket.rows[0][0] == 1);
        assert(bucket.rows[1][2] == 60);

        pushRow(11, 12, 13);
        assert(bucket.rows[2][0] == 11);
        assert(bucket.rows[2][1] == 12);
        assert(bucket.rows[2][2] == 13);

        bucket.lid = 99;
        assert(bucket.lid == 99);
        assert(bucket.rows[0][1] == 2);
        assert(bucket.rows[1][2] == 60);
        assert(bucket.rows[2][2] == 13);

        // pop again — back to 2 rows
        bucket.rows.pop();
        assert(bucket.rows.length == 2);
        assert(bucket.rows[0][0] == 1);
        assert(bucket.rows[1][1] == 5);

        // push twice more
        pushRow(20, 21, 22);
        pushRow(30, 31, 32);
        assert(bucket.rows.length == 4);
        assert(bucket.rows[2][2] == 22);
        assert(bucket.rows[3][2] == 32);
    }
}

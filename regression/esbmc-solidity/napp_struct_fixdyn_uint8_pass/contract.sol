// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

// Stress: struct with `uint8[3][]` member — fixed-inner dyn-outer of
// uint8. Exercises mixed-shape array nested in struct.
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

        // mutate slot 1 element 2
        bucket.rows[1][2] = 60;
        assert(bucket.rows[1][2] == 60);
        // others unchanged
        assert(bucket.rows[1][0] == 4);
        assert(bucket.rows[1][1] == 5);

        // pop — last row gone
        bucket.rows.pop();
        assert(bucket.rows.length == 2);
        assert(bucket.rows[0][0] == 1);
        assert(bucket.rows[1][2] == 60);

        // push new row
        pushRow(11, 12, 13);
        assert(bucket.rows.length == 3);
        assert(bucket.rows[2][0] == 11);
        assert(bucket.rows[2][1] == 12);
        assert(bucket.rows[2][2] == 13);

        // mutate lid
        bucket.lid = 99;
        assert(bucket.lid == 99);

        // earlier rows still intact
        assert(bucket.rows[0][0] == 1);
        assert(bucket.rows[0][1] == 2);
        assert(bucket.rows[0][2] == 3);
        assert(bucket.rows[1][0] == 4);
        assert(bucket.rows[1][1] == 5);
        assert(bucket.rows[1][2] == 60);
        assert(bucket.rows[2][2] == 13);
    }
}

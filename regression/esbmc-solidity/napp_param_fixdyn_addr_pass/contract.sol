// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

// Stress: nested array as memory function parameter with mixed shape
// `address[3][]` (fixed-inner-3, dyn-outer). Exercises memory layout
// for fixed-size address tuples carried across call boundaries.
contract C {
    uint256 internal seenRows;

    function checkRow(address[3][] memory rows, uint256 i,
                      address e0, address e1, address e2) internal pure {
        assert(rows[i][0] == e0);
        assert(rows[i][1] == e1);
        assert(rows[i][2] == e2);
    }

    function tally(address[3][] memory rows) internal returns (uint256) {
        // count rows whose middle slot is non-zero
        uint256 nz = 0;
        for (uint256 i = 0; i < rows.length; i++) {
            if (rows[i][1] != address(0)) {
                nz += 1;
            }
        }
        seenRows = rows.length;
        return nz;
    }

    function run() external {
        address a = address(0x1);
        address b = address(0x2);
        address c = address(0x3);
        address z = address(0);

        address[3][] memory rows = new address[3][](4);
        rows[0][0] = a; rows[0][1] = b; rows[0][2] = c;
        rows[1][0] = a; rows[1][1] = z; rows[1][2] = c;
        rows[2][0] = b; rows[2][1] = c; rows[2][2] = a;
        rows[3][0] = z; rows[3][1] = z; rows[3][2] = z;

        assert(rows.length == 4);
        checkRow(rows, 0, a, b, c);
        checkRow(rows, 1, a, z, c);
        checkRow(rows, 2, b, c, a);
        checkRow(rows, 3, z, z, z);

        // row 0 and row 2 have non-zero middle; row 1 and 3 have zero
        uint256 nz = tally(rows);
        assert(nz == 2);
        assert(seenRows == 4);

        // mutate row 1 middle to non-zero, re-tally
        rows[1][1] = a;
        nz = tally(rows);
        assert(nz == 3);
        assert(seenRows == 4);

        // mutate row 3 middle to non-zero, re-tally
        rows[3][1] = b;
        nz = tally(rows);
        assert(nz == 4);
        assert(seenRows == 4);

        // mutate row 0 to zero — count returns to 3
        rows[0][1] = z;
        nz = tally(rows);
        assert(nz == 3);

        // row 0 changed, row 2 unaffected
        assert(rows[0][1] == z);
        checkRow(rows, 2, b, c, a);
    }
}
// KNOWNBUG: the inductive step cannot prove this without the inductive
// invariants the local k-induction inferred (loop-entry invariants,
// counted-loop skip, per-field havoc), which upstream's rewrite dropped.

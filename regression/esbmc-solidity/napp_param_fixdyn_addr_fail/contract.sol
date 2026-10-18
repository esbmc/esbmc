// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

// Dual-fail: same address[3][] param harness, flipped non-zero count
// expectation after first tally call.
contract C {
    uint256 internal seenRows;

    function checkRow(address[3][] memory rows, uint256 i,
                      address e0, address e1, address e2) internal pure {
        assert(rows[i][0] == e0);
        assert(rows[i][1] == e1);
        assert(rows[i][2] == e2);
    }

    function tally(address[3][] memory rows) internal returns (uint256) {
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

        uint256 nz = tally(rows);
        // FLIPPED: non-zero middles are rows 0 and 2, count is 2 not 4
        assert(nz == 4);
        assert(seenRows == 4);

        rows[1][1] = a;
        nz = tally(rows);
        assert(nz == 3);

        rows[3][1] = b;
        nz = tally(rows);
        assert(nz == 4);

        // mutate row 0 to zero — count returns to 3
        rows[0][1] = z;
        nz = tally(rows);
        assert(nz == 3);
        assert(rows[0][1] == z);
        checkRow(rows, 2, b, c, a);

        // restore — count returns to 4
        rows[0][1] = b;
        nz = tally(rows);
        assert(nz == 4);
    }
}

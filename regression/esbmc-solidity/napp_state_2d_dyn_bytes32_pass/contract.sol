// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

// Stress: 2D dynamic array of `bytes32` as state variable.
// `bytes32` lowers to BytesStatic in ESBMC's model — exercises
// fixed-bytes element packing inside a dynamic array nested in a
// dynamic array.
contract C {
    bytes32[][] internal hashes;

    function pushRow() internal {
        hashes.push();
    }

    function pushHash(uint256 r, bytes32 h) internal {
        hashes[r].push(h);
    }

    function run() external {
        bytes32 h1 = bytes32(uint256(0xAA));
        bytes32 h2 = bytes32(uint256(0xBB));
        bytes32 h3 = bytes32(uint256(0xCC));
        bytes32 h4 = bytes32(uint256(0xDD));

        assert(hashes.length == 0);

        pushRow();
        pushRow();
        pushRow();
        assert(hashes.length == 3);
        assert(hashes[0].length == 0);
        assert(hashes[1].length == 0);
        assert(hashes[2].length == 0);

        pushHash(0, h1);
        pushHash(0, h2);
        pushHash(0, h3);
        assert(hashes[0].length == 3);
        assert(hashes[0][0] == h1);
        assert(hashes[0][1] == h2);
        assert(hashes[0][2] == h3);

        pushHash(1, h4);
        pushHash(1, h1);
        assert(hashes[1].length == 2);
        assert(hashes[1][0] == h4);
        assert(hashes[1][1] == h1);

        // row 2 stays empty across operations on rows 0, 1
        assert(hashes[2].length == 0);

        // pop two from row 0
        hashes[0].pop();
        hashes[0].pop();
        assert(hashes[0].length == 1);
        assert(hashes[0][0] == h1);

        // re-push different bytes32 values
        pushHash(0, h4);
        pushHash(0, h2);
        assert(hashes[0].length == 3);
        assert(hashes[0][1] == h4);
        assert(hashes[0][2] == h2);

        // outer pop the empty row 2
        hashes.pop();
        assert(hashes.length == 2);

        // remaining content unchanged
        assert(hashes[1].length == 2);
        assert(hashes[1][0] == h4);
        assert(hashes[0][0] == h1);
    }
}

// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

// Dual-fail: identical 3D harness with a flipped invariant on the
// re-pushed pair after pop. Same sequence as the PASS variant up to
// the flipped check, plus full follow-up assertions.
contract C {
    int256[2][][3] internal cube;

    function fillPair(uint256 outer, int256 a, int256 b) internal {
        int256[2] memory pair;
        pair[0] = a;
        pair[1] = b;
        cube[outer].push(pair);
    }

    function run() external {
        assert(cube[0].length == 0);
        assert(cube[1].length == 0);
        assert(cube[2].length == 0);

        fillPair(0, 1, -1);
        fillPair(0, 2, -2);
        fillPair(0, 3, -3);
        assert(cube[0].length == 3);
        assert(cube[0][0][0] == 1);
        assert(cube[0][0][1] == -1);
        assert(cube[0][1][0] == 2);
        assert(cube[0][1][1] == -2);
        assert(cube[0][2][0] == 3);
        assert(cube[0][2][1] == -3);

        fillPair(1, 100, -100);
        fillPair(1, 200, -200);
        assert(cube[1].length == 2);
        assert(cube[1][0][0] == 100);
        assert(cube[1][0][1] == -100);
        assert(cube[1][1][0] == 200);
        assert(cube[1][1][1] == -200);

        assert(cube[2].length == 0);

        cube[0].pop();
        assert(cube[0].length == 2);
        assert(cube[0][1][0] == 2);

        fillPair(0, 999, -999);
        assert(cube[0].length == 3);
        // FLIPPED: after re-push, cube[0][2][0] is 999, not 3
        assert(cube[0][2][0] == 3);
        assert(cube[0][2][1] == -999);

        fillPair(2, 7, -7);
        assert(cube[2].length == 1);
        assert(cube[2][0][0] == 7);
        assert(cube[2][0][1] == -7);

        assert(cube[0].length == 3);
        assert(cube[1].length == 2);
        assert(cube[1][0][1] == -100);
        assert(cube[0][0][0] == 1);

        // pop cube[2] back to empty
        cube[2].pop();
        assert(cube[2].length == 0);

        // re-fill cube[2] with two pairs
        fillPair(2, 5, -5);
        fillPair(2, 6, -6);
        assert(cube[2].length == 2);
        assert(cube[2][0][0] == 5);
        assert(cube[2][1][1] == -6);
    }
}

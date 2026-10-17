// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

// Stress: 3D state-var array with mixed shape `int256[2][][3]`.
// Outer is fixed-size 3, middle is dynamic, inner is fixed-size 2.
// Tests push/pop on the middle dynamic dimension across multiple
// outer slots, with signed-integer element values.
contract C {
    int256[2][][3] internal cube;

    function fillPair(uint256 outer, int256 a, int256 b) internal {
        int256[2] memory pair;
        pair[0] = a;
        pair[1] = b;
        cube[outer].push(pair);
    }

    function run() external {
        // outer is fixed-3 — every slot is an empty dyn array initially
        assert(cube[0].length == 0);
        assert(cube[1].length == 0);
        assert(cube[2].length == 0);

        // populate cube[0] with three pairs
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

        // populate cube[1] with two pairs
        fillPair(1, 100, -100);
        fillPair(1, 200, -200);
        assert(cube[1].length == 2);
        assert(cube[1][0][0] == 100);
        assert(cube[1][1][1] == -200);

        // cube[2] still empty
        assert(cube[2].length == 0);

        // pop from cube[0]
        cube[0].pop();
        assert(cube[0].length == 2);
        assert(cube[0][1][0] == 2);
        assert(cube[0][1][1] == -2);

        // re-push different pair after pop
        fillPair(0, 999, -999);
        assert(cube[0].length == 3);
        assert(cube[0][2][0] == 999);
        assert(cube[0][2][1] == -999);

        // populate cube[2] last
        fillPair(2, 7, -7);
        assert(cube[2].length == 1);
        assert(cube[2][0][0] == 7);
        assert(cube[2][0][1] == -7);

        // earlier slot data unaffected
        assert(cube[0].length == 3);
        assert(cube[1].length == 2);
        assert(cube[1][0][1] == -100);
        assert(cube[0][0][0] == 1);
    }
}

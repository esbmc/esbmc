// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

// Stress test: 2D dynamic array of `bool` as a state variable.
// Exercises: outer push/pop, inner push/pop, length reads at every
// step, indexed reads after both push and pop, and re-push after pop
// to cover slot reuse.
contract C {
    bool[][] internal flags;

    function pushRow() internal {
        flags.push();
    }

    function pushVal(uint256 r, bool v) internal {
        flags[r].push(v);
    }

    function popInner(uint256 r) internal {
        flags[r].pop();
    }

    function run() external {
        // initially empty
        assert(flags.length == 0);

        // push three outer rows
        pushRow();
        pushRow();
        pushRow();
        assert(flags.length == 3);
        assert(flags[0].length == 0);
        assert(flags[1].length == 0);
        assert(flags[2].length == 0);

        // populate row 0 with alternating values
        pushVal(0, true);
        pushVal(0, false);
        pushVal(0, true);
        pushVal(0, false);
        assert(flags[0].length == 4);
        assert(flags[0][0] == true);
        assert(flags[0][1] == false);
        assert(flags[0][2] == true);
        assert(flags[0][3] == false);

        // populate row 1
        pushVal(1, false);
        pushVal(1, true);
        assert(flags[1].length == 2);
        assert(flags[1][0] == false);
        assert(flags[1][1] == true);

        // row 2 stays empty
        assert(flags[2].length == 0);

        // pop two values from row 0
        popInner(0);
        popInner(0);
        assert(flags[0].length == 2);
        assert(flags[0][0] == true);
        assert(flags[0][1] == false);

        // re-push after pop covers slot reuse
        pushVal(0, true);
        pushVal(0, true);
        assert(flags[0].length == 4);
        assert(flags[0][2] == true);
        assert(flags[0][3] == true);

        // pop the empty outer row (row 2)
        flags.pop();
        assert(flags.length == 2);

        // remaining rows unchanged
        assert(flags[0].length == 4);
        assert(flags[1].length == 2);
        assert(flags[1][1] == true);
        assert(flags[0][0] == true);

        // restore state for inductive harness — dispatcher loop calls
        // run() repeatedly; without restore the second call's entry
        // assertion `flags.length == 0` would fail
        popInner(0); popInner(0); popInner(0); popInner(0);
        popInner(1); popInner(1);
        flags.pop(); flags.pop();
        assert(flags.length == 0);
    }
}

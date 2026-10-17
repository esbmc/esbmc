// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

// Dual-fail of napp_state_2d_dyn_bool_pass: identical setup, single
// flipped invariant. After two inner pops, flags[0].length is 2; the
// FAIL variant asserts 3 to force a counter-example.
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
        assert(flags.length == 0);

        pushRow();
        pushRow();
        pushRow();
        assert(flags.length == 3);
        assert(flags[0].length == 0);
        assert(flags[1].length == 0);
        assert(flags[2].length == 0);

        pushVal(0, true);
        pushVal(0, false);
        pushVal(0, true);
        pushVal(0, false);
        assert(flags[0].length == 4);
        assert(flags[0][0] == true);
        assert(flags[0][1] == false);
        assert(flags[0][2] == true);
        assert(flags[0][3] == false);

        pushVal(1, false);
        pushVal(1, true);
        assert(flags[1].length == 2);
        assert(flags[1][0] == false);
        assert(flags[1][1] == true);

        assert(flags[2].length == 0);

        popInner(0);
        popInner(0);
        // FLIPPED: actual length is 2 after two pops, not 3.
        assert(flags[0].length == 3);
        assert(flags[0][0] == true);
        assert(flags[0][1] == false);

        pushVal(0, true);
        pushVal(0, true);
        assert(flags[0].length == 4);
        assert(flags[0][2] == true);
        assert(flags[0][3] == true);

        flags.pop();
        assert(flags.length == 2);

        assert(flags[0].length == 4);
        assert(flags[1].length == 2);
        assert(flags[1][1] == true);
        assert(flags[0][0] == true);
    }
}

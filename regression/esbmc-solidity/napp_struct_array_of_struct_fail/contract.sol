// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

// Dual-fail: same struct-of-struct-array harness with flipped tag
// invariant after re-push.
contract C {
    struct Inner {
        bytes32 tag;
        bool active;
    }

    struct Outer {
        uint256 epoch;
        Inner[][] g;
    }

    Outer internal outer;

    function pushRow() internal {
        outer.g.push();
    }

    function pushI(uint256 r, bytes32 t, bool a) internal {
        Inner memory i;
        i.tag = t;
        i.active = a;
        outer.g[r].push(i);
    }

    function run() external {
        outer.epoch = 1;
        bytes32 t1 = bytes32(uint256(0x11));
        bytes32 t2 = bytes32(uint256(0x22));
        bytes32 t3 = bytes32(uint256(0x33));

        assert(outer.epoch == 1);
        assert(outer.g.length == 0);

        pushRow();
        pushRow();
        assert(outer.g.length == 2);

        pushI(0, t1, true);
        pushI(0, t2, false);
        assert(outer.g[0].length == 2);
        assert(outer.g[0][0].tag == t1);
        assert(outer.g[0][0].active == true);
        assert(outer.g[0][1].tag == t2);
        assert(outer.g[0][1].active == false);

        pushI(1, t3, true);
        assert(outer.g[1].length == 1);
        assert(outer.g[1][0].tag == t3);

        outer.g[0][0].active = false;
        outer.epoch = 2;
        assert(outer.g[0][0].active == false);
        assert(outer.epoch == 2);

        outer.g[0].pop();
        assert(outer.g[0].length == 1);
        assert(outer.g[0][0].tag == t1);

        pushI(0, t3, true);
        // FLIPPED: outer.g[0][1].tag is t3 after re-push, not t2
        assert(outer.g[0][1].tag == t2);
        assert(outer.epoch == 2);
        assert(outer.g[0][1].active == true);
    }
}

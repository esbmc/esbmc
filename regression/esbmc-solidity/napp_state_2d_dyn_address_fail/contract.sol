// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

// Dual-fail of napp_state_2d_dyn_address_pass: identical setup with
// one flipped invariant — after re-pushing a4 into row 0 slot 2, the
// FAIL variant asserts owners[0][2] == a3 (the value before pop+push).
contract C {
    address[][] internal owners;

    function pushRow() internal {
        owners.push();
    }

    function pushAddr(uint256 r, address a) internal {
        owners[r].push(a);
    }

    function run() external {
        address a1 = address(0x1111);
        address a2 = address(0x2222);
        address a3 = address(0x3333);
        address a4 = address(0x4444);

        assert(owners.length == 0);

        pushRow();
        pushRow();
        assert(owners.length == 2);
        assert(owners[0].length == 0);
        assert(owners[1].length == 0);

        pushAddr(0, a1);
        pushAddr(0, a2);
        pushAddr(0, a3);
        assert(owners[0].length == 3);
        assert(owners[0][0] == a1);
        assert(owners[0][1] == a2);
        assert(owners[0][2] == a3);

        pushAddr(1, a4);
        assert(owners[1].length == 1);
        assert(owners[1][0] == a4);

        owners[0].pop();
        assert(owners[0].length == 2);
        assert(owners[0][0] == a1);
        assert(owners[0][1] == a2);
        assert(owners[1].length == 1);

        pushAddr(0, a4);
        assert(owners[0].length == 3);
        // FLIPPED: actual value is a4 after re-push, not a3
        assert(owners[0][2] == a3);
        assert(owners[0][0] == a1);
        assert(owners[0][1] == a2);

        pushRow();
        assert(owners.length == 3);
        pushAddr(2, a1);
        pushAddr(2, a2);
        assert(owners[2].length == 2);
        assert(owners[2][0] == a1);
        assert(owners[2][1] == a2);

        owners.pop();
        assert(owners.length == 2);
        assert(owners[0].length == 3);
        assert(owners[1].length == 1);
        assert(owners[1][0] == a4);
        assert(owners[0][0] == a1);
        assert(owners[0][1] == a2);
    }
}

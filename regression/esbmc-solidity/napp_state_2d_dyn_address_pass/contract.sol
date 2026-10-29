// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

// Stress: 2D dynamic array of `address` as state variable.
// Exercises outer/inner push, pop, length, indexed read, mid-sequence
// pop+re-push, and address equality after store.
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

        // pop tail of row 0; row 1 untouched
        owners[0].pop();
        assert(owners[0].length == 2);
        assert(owners[0][0] == a1);
        assert(owners[0][1] == a2);
        assert(owners[1].length == 1);
        assert(owners[1][0] == a4);

        // re-push different address
        pushAddr(0, a4);
        assert(owners[0].length == 3);
        assert(owners[0][2] == a4);

        // push another outer row, populate, then outer-pop it
        pushRow();
        assert(owners.length == 3);
        pushAddr(2, a1);
        pushAddr(2, a2);
        assert(owners[2].length == 2);
        assert(owners[2][0] == a1);
        assert(owners[2][1] == a2);

        owners.pop();
        assert(owners.length == 2);

        // earlier rows preserved
        assert(owners[0].length == 3);
        assert(owners[1].length == 1);
        assert(owners[0][2] == a4);
        assert(owners[1][0] == a4);
    }
}

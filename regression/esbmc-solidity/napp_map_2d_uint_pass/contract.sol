// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

// Stress: mapping with nested-array value `mapping(address =>
// uint256[][])`. Exercises mapping-keyed access to a 2D dynamic
// array, with cross-key isolation.
contract C {
    mapping(address => uint256[][]) internal book;

    function pushRow(address k) internal {
        book[k].push();
    }

    function pushVal(address k, uint256 r, uint256 v) internal {
        book[k][r].push(v);
    }

    function run() external {
        address ka = address(0xA);
        address kb = address(0xB);

        // initially empty for both keys
        assert(book[ka].length == 0);
        assert(book[kb].length == 0);

        // build under ka
        pushRow(ka);
        pushRow(ka);
        assert(book[ka].length == 2);

        pushVal(ka, 0, 1);
        pushVal(ka, 0, 2);
        pushVal(ka, 0, 3);
        assert(book[ka][0].length == 3);
        assert(book[ka][0][0] == 1);
        assert(book[ka][0][1] == 2);
        assert(book[ka][0][2] == 3);

        pushVal(ka, 1, 100);
        assert(book[ka][1].length == 1);
        assert(book[ka][1][0] == 100);

        // ka pushes must not leak to kb
        assert(book[kb].length == 0);

        // build under kb
        pushRow(kb);
        assert(book[kb].length == 1);
        pushVal(kb, 0, 999);
        pushVal(kb, 0, 888);
        assert(book[kb][0].length == 2);
        assert(book[kb][0][0] == 999);
        assert(book[kb][0][1] == 888);

        // ka data unaffected by kb push
        assert(book[ka].length == 2);
        assert(book[ka][0].length == 3);
        assert(book[ka][0][2] == 3);

        // pop under ka
        book[ka][0].pop();
        assert(book[ka][0].length == 2);
        assert(book[ka][0][1] == 2);

        // re-push under ka, kb unaffected
        pushVal(ka, 0, 7);
        assert(book[ka][0].length == 3);
        assert(book[ka][0][2] == 7);
        assert(book[kb][0].length == 2);
        assert(book[kb][0][1] == 888);
    }
}

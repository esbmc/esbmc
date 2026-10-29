// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

// Dual-fail: same mapping(address => uint256[][]) harness with
// flipped cross-key isolation invariant — asserts ka leaked to kb.
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

        assert(book[ka].length == 0);
        assert(book[kb].length == 0);

        pushRow(ka);
        pushRow(ka);
        assert(book[ka].length == 2);

        pushVal(ka, 0, 1);
        pushVal(ka, 0, 2);
        pushVal(ka, 0, 3);
        assert(book[ka][0].length == 3);
        assert(book[ka][0][0] == 1);
        assert(book[ka][0][2] == 3);

        pushVal(ka, 1, 100);
        assert(book[ka][1].length == 1);
        assert(book[ka][1][0] == 100);

        // FLIPPED: kb has no rows pushed, length must be 0 not 2
        assert(book[kb].length == 2);

        pushRow(kb);
        assert(book[kb].length == 1);
        pushVal(kb, 0, 999);
        pushVal(kb, 0, 888);
        assert(book[kb][0].length == 2);
        assert(book[kb][0][0] == 999);
        assert(book[kb][0][1] == 888);

        assert(book[ka].length == 2);
        assert(book[ka][0].length == 3);
        assert(book[ka][0][2] == 3);

        book[ka][0].pop();
        assert(book[ka][0].length == 2);
        assert(book[ka][0][1] == 2);

        pushVal(ka, 0, 7);
        assert(book[ka][0].length == 3);
        assert(book[ka][0][2] == 7);
        assert(book[ka][0][0] == 1);
        assert(book[ka][0][1] == 2);

        // re-mutate under kb to confirm independence
        book[kb][0].pop();
        assert(book[kb][0].length == 1);
        assert(book[kb][0][0] == 999);
        pushVal(kb, 0, 555);
        assert(book[kb][0].length == 2);
        assert(book[kb][0][1] == 555);

        // ka again unaffected by kb pop+push
        assert(book[ka].length == 2);
        assert(book[ka][0].length == 3);
        assert(book[ka][1].length == 1);
    }
}

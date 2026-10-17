// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

// Dual-fail: same mapping(addr => S[][]) harness with flipped cross-
// key isolation invariant.
contract C {
    struct S {
        uint256 weight;
        bytes32 label;
    }

    mapping(address => S[][]) internal store;

    function pushRow(address k) internal {
        store[k].push();
    }

    function pushS(address k, uint256 r, uint256 w, bytes32 l) internal {
        S memory s;
        s.weight = w;
        s.label = l;
        store[k][r].push(s);
    }

    function run() external {
        address ka = address(0xA);
        address kb = address(0xB);
        bytes32 la = bytes32(uint256(0xAA));
        bytes32 lb = bytes32(uint256(0xBB));
        bytes32 lc = bytes32(uint256(0xCC));

        assert(store[ka].length == 0);
        assert(store[kb].length == 0);

        pushRow(ka);
        pushRow(ka);
        assert(store[ka].length == 2);

        pushS(ka, 0, 10, la);
        pushS(ka, 0, 20, lb);
        assert(store[ka][0].length == 2);
        assert(store[ka][0][0].weight == 10);
        assert(store[ka][0][0].label == la);
        assert(store[ka][0][1].weight == 20);

        pushS(ka, 1, 30, lc);
        assert(store[ka][1].length == 1);
        assert(store[ka][1][0].weight == 30);

        // FLIPPED: kb has no rows pushed; length is 0 not 1
        assert(store[kb].length == 1);

        pushRow(kb);
        pushS(kb, 0, 999, la);
        assert(store[kb][0].length == 1);
        assert(store[kb][0][0].weight == 999);
        assert(store[kb][0][0].label == la);

        assert(store[ka][0][0].weight == 10);
        assert(store[ka][1][0].label == lc);

        // mutate field under ka
        store[ka][0][1].weight = 25;
        assert(store[ka][0][1].weight == 25);
        assert(store[ka][0][1].label == lb);

        // pop+re-push under ka
        store[ka][0].pop();
        assert(store[ka][0].length == 1);
        pushS(ka, 0, 88, lc);
        assert(store[ka][0].length == 2);
        assert(store[ka][0][1].weight == 88);
    }
}

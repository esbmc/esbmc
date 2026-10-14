// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

// Stress: nested array as a function RETURN value — `uint256[][]
// memory`. Tests that the callee can build a fresh nested-dynamic
// array in memory, return it, and the caller can read length and
// elements consistently.
contract C {
    uint256 internal lastTotal;

    function build(uint256 base) internal pure returns (uint256[][] memory) {
        uint256[][] memory g = new uint256[][](3);
        g[0] = new uint256[](2);
        g[0][0] = base;
        g[0][1] = base + 1;
        g[1] = new uint256[](3);
        g[1][0] = base + 10;
        g[1][1] = base + 20;
        g[1][2] = base + 30;
        g[2] = new uint256[](1);
        g[2][0] = base + 100;
        return g;
    }

    function totalize(uint256[][] memory g) internal pure returns (uint256) {
        uint256 t = 0;
        for (uint256 i = 0; i < g.length; i++) {
            for (uint256 j = 0; j < g[i].length; j++) {
                t += g[i][j];
            }
        }
        return t;
    }

    function run() external {
        // first call: base = 0
        uint256[][] memory g0 = build(0);
        assert(g0.length == 3);
        assert(g0[0].length == 2);
        assert(g0[1].length == 3);
        assert(g0[2].length == 1);
        assert(g0[0][0] == 0);
        assert(g0[0][1] == 1);
        assert(g0[1][0] == 10);
        assert(g0[1][2] == 30);
        assert(g0[2][0] == 100);

        uint256 t0 = totalize(g0);
        // 0 + 1 + 10 + 20 + 30 + 100 = 161
        assert(t0 == 161);

        // second call: base = 5
        uint256[][] memory g1 = build(5);
        assert(g1.length == 3);
        assert(g1[0][0] == 5);
        assert(g1[1][2] == 35);
        assert(g1[2][0] == 105);

        uint256 t1 = totalize(g1);
        // (5 + 6) + (15 + 25 + 35) + 105 = 191
        assert(t1 == 191);

        // mutate g0 in caller — g1 must be unaffected
        g0[1][1] = 999;
        assert(g0[1][1] == 999);
        assert(g1[1][1] == 25);

        lastTotal = t1;
        assert(lastTotal == 191);
    }
}
// KNOWNBUG: the inductive step cannot prove this without the inductive
// invariants the local k-induction inferred (loop-entry invariants,
// counted-loop skip, per-field havoc), which upstream's rewrite dropped.

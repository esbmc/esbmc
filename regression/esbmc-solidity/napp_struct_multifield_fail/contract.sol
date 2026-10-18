// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

// Dual-fail: identical multi-field struct harness, flipped invariant
// on the bool inner length after pop+push.
contract C {
    struct Trio {
        address[][] addrs;
        bytes32[3][] hashes;
        bool[][2] flags;
    }

    Trio internal trio;

    function pushAddrRow() internal {
        trio.addrs.push();
    }

    function pushAddr(uint256 r, address a) internal {
        trio.addrs[r].push(a);
    }

    function pushHashes(bytes32 a, bytes32 b, bytes32 c) internal {
        bytes32[3] memory row;
        row[0] = a;
        row[1] = b;
        row[2] = c;
        trio.hashes.push(row);
    }

    function pushFlag(uint256 outer, bool v) internal {
        trio.flags[outer].push(v);
    }

    function run() external {
        pushAddrRow();
        pushAddr(0, address(0xAA));
        pushAddr(0, address(0xBB));
        assert(trio.addrs.length == 1);
        assert(trio.addrs[0].length == 2);

        pushHashes(bytes32(uint256(1)), bytes32(uint256(2)), bytes32(uint256(3)));
        pushHashes(bytes32(uint256(4)), bytes32(uint256(5)), bytes32(uint256(6)));
        assert(trio.hashes.length == 2);

        pushFlag(0, true);
        pushFlag(0, false);
        pushFlag(1, true);
        assert(trio.flags[0].length == 2);
        assert(trio.flags[1].length == 1);

        trio.addrs[0].pop();
        trio.hashes.pop();
        trio.flags[0].pop();

        pushAddr(0, address(0xCC));
        pushHashes(bytes32(uint256(9)), bytes32(uint256(9)), bytes32(uint256(9)));
        pushFlag(0, false);
        pushFlag(1, false);

        // FLIPPED: trio.flags[1] has length 2 after second push, not 3
        assert(trio.flags[1].length == 3);
        assert(trio.addrs[0].length == 2);
        assert(trio.addrs[0][1] == address(0xCC));
        assert(trio.hashes.length == 2);
        assert(trio.hashes[1][0] == bytes32(uint256(9)));
        assert(trio.hashes[1][1] == bytes32(uint256(9)));
        assert(trio.hashes[1][2] == bytes32(uint256(9)));
        assert(trio.flags[0].length == 2);
        assert(trio.flags[0][0] == true);
        assert(trio.flags[0][1] == false);
    }
}

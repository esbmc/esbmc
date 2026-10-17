// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

// Stress: struct with three nested-array fields of differing shapes
// and element types — `address[][]`, `bytes32[3][]`, `bool[][2]`.
// Exercises mixed shapes co-existing inside a single storage struct.
contract C {
    struct Trio {
        address[][] addrs;       // 2D dyn-dyn
        bytes32[3][] hashes;     // fixed-inner dyn-outer
        bool[][2] flags;         // dyn-inner fixed-outer
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
        // address[][]
        pushAddrRow();
        pushAddr(0, address(0xAA));
        pushAddr(0, address(0xBB));
        assert(trio.addrs.length == 1);
        assert(trio.addrs[0].length == 2);
        assert(trio.addrs[0][0] == address(0xAA));
        assert(trio.addrs[0][1] == address(0xBB));

        // bytes32[3][]
        pushHashes(bytes32(uint256(1)), bytes32(uint256(2)), bytes32(uint256(3)));
        pushHashes(bytes32(uint256(4)), bytes32(uint256(5)), bytes32(uint256(6)));
        assert(trio.hashes.length == 2);
        assert(trio.hashes[0][0] == bytes32(uint256(1)));
        assert(trio.hashes[1][2] == bytes32(uint256(6)));

        // bool[][2]
        pushFlag(0, true);
        pushFlag(0, false);
        pushFlag(1, true);
        assert(trio.flags[0].length == 2);
        assert(trio.flags[1].length == 1);
        assert(trio.flags[0][0] == true);
        assert(trio.flags[0][1] == false);
        assert(trio.flags[1][0] == true);

        // pop from each shape
        trio.addrs[0].pop();
        assert(trio.addrs[0].length == 1);
        assert(trio.addrs[0][0] == address(0xAA));

        trio.hashes.pop();
        assert(trio.hashes.length == 1);
        assert(trio.hashes[0][1] == bytes32(uint256(2)));

        trio.flags[0].pop();
        assert(trio.flags[0].length == 1);
        assert(trio.flags[0][0] == true);

        // re-push to verify slot reuse for each field
        pushAddr(0, address(0xCC));
        pushHashes(bytes32(uint256(9)), bytes32(uint256(9)), bytes32(uint256(9)));
        pushFlag(0, false);
        pushFlag(1, false);
        assert(trio.addrs[0].length == 2);
        assert(trio.addrs[0][1] == address(0xCC));
        assert(trio.hashes.length == 2);
        assert(trio.hashes[1][1] == bytes32(uint256(9)));
        assert(trio.flags[0].length == 2);
        assert(trio.flags[1].length == 2);
    }
}

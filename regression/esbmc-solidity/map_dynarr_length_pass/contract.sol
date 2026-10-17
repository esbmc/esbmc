// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

// Length after N pushes: three pushes must yield length == 3.
//
// The `require(length == 0)` pin handles the `--bound` harness loop:
// the non-deterministic Extcall dispatcher may invoke `test()` multiple
// times, so without the pin, `length == 3 * N`. Pinning prunes every
// path where a prior call already pushed, leaving the exact-three-push
// path on which the assertion holds.
contract C {
    mapping(address => uint256[]) m;
    function test() public {
        require(m[address(0x1)].length == 0);
        m[address(0x1)].push(1);
        m[address(0x1)].push(2);
        m[address(0x1)].push(3);
        assert(m[address(0x1)].length == 3);
    }
}

// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

// Write-through semantics for mapping(K => V[]).push(x).
// After the push, m[a][0] must equal the pushed value.
contract C {
    mapping(address => uint256[]) m;
    function test() public {
        m[address(0x1)].push(42);
        assert(m[address(0x1)][0] == 42);
    }
}

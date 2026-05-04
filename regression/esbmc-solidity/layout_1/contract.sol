// SPDX-License-Identifier: GPL-3.0
pragma solidity ^0.8.29;

// Test custom storage layout (Solidity 0.8.29+).
// ESBMC does not model EVM storage slots, so the layout specifier
// is ignored — but the AST must parse without errors.
contract C layout at 0xAAAA + 0x11 {
    uint[3] x; // Occupies slots 0xAABB..0xAABD

    function test() public {
        x[0] = 10;
        x[1] = 20;
        x[2] = 30;
        assert(x[0] + x[1] + x[2] == 60);
    }
}

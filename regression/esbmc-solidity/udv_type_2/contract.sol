// SPDX-License-Identifier: GPL-3.0
pragma solidity ^0.8.19;

// Test user-defined value types — wrap/unwrap preserves value (--bound mode).
type Price is uint16;

contract UDVTestFail {
    function test() public pure {
        Price p = Price.wrap(42);
        uint16 raw = Price.unwrap(p);
        // FAILS: raw is 42, not 0
        assert(raw == 0);
    }
}

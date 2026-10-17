// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

// T1.2 Stage S0 — PASS companion locking in that contracts which explicitly
// precondition the bytes length via `require` continue to verify after the
// Stage S1 clamp removal. Today and post-fix: SUCCESSFUL.
contract H {
    function f(bytes calldata d) external pure {
        require(d.length >= 32 && d.length <= 1024);
        assert(d.length >= 32);
    }
}

// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

// T2.2 — positive regression coverage for try/catch where the call has
// no state mutation. Catch arm trivially sees pre-call state because
// there's nothing to roll back. Locks in that the rollback model
// (B1 + SSA branching) does not regress the simplest catch path.
contract Target {
    function read() external pure returns (uint256) {
        return 42;
    }
}

contract H {
    Target target;

    constructor() {
        target = new Target();
    }

    function check() external view {
        try target.read() returns (uint256) {
            assert(true);
        } catch {
            assert(true);
        }
    }
}

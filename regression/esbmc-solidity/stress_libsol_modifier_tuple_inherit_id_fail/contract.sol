// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

// Dual to pass: the same tuple-returning, modifier-wrapped function
// must reach symex (frontend must not abort on tuple_instance$0
// lookup). An unconditional false assert anchored ahead of the call
// guarantees the regression produces VERIFICATION FAILED rather than
// being sliced to zero VCCs.

contract C {
    modifier nonReentrant() {
        _;
    }

    function pair(uint256 a) external pure nonReentrant returns (uint256, uint256) {
        return (a, a + 1);
    }

    function go() external view {
        assert(false);
        // Even though the assert above is unreachable, the frontend
        // still lowers the statement below, so the tuple-returning
        // modifier-wrapped path is exercised at Converting time.
        this.pair(7);
    }
}

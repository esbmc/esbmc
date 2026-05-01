// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.8.0;

// Negative counterpart of ext_call_trusted_1. After e.setX(42), x is 42,
// so checkX(41) must fire its internal assertion. ESBMC's --bound mode
// must report VERIFICATION FAILED rather than vacuously succeeding,
// proving that inter-contract state tracking is real (not a nondet
// over-approximation that accepts everything).

contract Ext {
    uint x;

    function setX(uint _x) public { x = _x; }

    function checkX(uint _expected) public view {
        assert(x == _expected);
    }
}

contract MyContract {
    Ext e = new Ext();

    function callExt() public {
        e.setX(42);
        // Intentionally wrong expected value.
        e.checkX(41);
    }
}

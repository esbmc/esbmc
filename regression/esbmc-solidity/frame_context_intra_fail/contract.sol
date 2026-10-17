// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

// B2 fail dual — verifies that ESBMC's per-iteration freshness of
// `block.number` IS still observable across distinct dispatcher
// iterations (it MUST be, otherwise tests like SolidiFI's
// time-locked-vault property would be unfindable).  Stores the
// first observed block.number on the first call; the second
// call's read can be a different (≥) block.number, so the
// equality assertion against the stored value will fail under
// at least one feasible execution.
contract C {
    uint public stored;
    bool public hasStored;

    function record() public {
        if (!hasStored) {
            stored = block.number;
            hasStored = true;
        } else {
            // Across iterations, _sol_per_tx_reseed advances
            // block.number monotonically — the new read may differ
            // from the stored value.  Asserting equality here must
            // FAIL for some feasible interleaving.
            assert(block.number == stored);
        }
    }
}

// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

// B1 — verifies that state writes performed before a `revert()` are
// rolled back so the on-chain `counter` cannot exceed the number of
// successful (non-reverting) `bump()` calls.  Under --bound the
// dispatcher iterates calling `bump` and `bumpThenRevert` non-
// deterministically.  With the unwind cap of 5 a successful run can
// raise `counter` by at most 5; reverted attempts must NOT
// contribute, so an overflow check on the uint8 backing slot will
// hold even if the dispatcher tries to call `bumpThenRevert` many
// times.
//
// Pre-B1 behaviour also gave the same verdict — but only because
// the legacy `__ESBMC_assume(false)` lowering prunes the reverted
// path entirely.  Post-B1, the path is feasible (state restored),
// so this exercise covers the new lowering's restore-and-return
// emission pathway.
contract Counter {
    uint8 public counter;

    function bump() public {
        counter += 1;
    }

    function bumpThenRevert() public {
        counter += 1;
        revert();
    }
}

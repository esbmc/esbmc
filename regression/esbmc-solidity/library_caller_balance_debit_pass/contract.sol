// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

// KNOWNBUG: Library transfers used to skip the caller contract's
// $balance debit entirely — libraries don't own a $balance slot and
// the enclosing contract was run-time data the frontend couldn't pin.
// After Stage 2 (ambient `_ESBMC_enclosing_contract_this` set at
// every contract-method entry, dispatched through a pointer-match
// helper `_ESBMC_enclosing_debit`), library transfers debit the
// enclosing contract's $balance even though the debit emitter runs
// inside the library body.
//
// Auto-dispatch entry: `--contract A` runs A.run() on the static
// singleton `_ESBMC_Object_A`, so `address(this)` matches the
// singleton pointer the debit helper dispatches on.  Once Stage 2
// lands, the assertion below holds (balance decreased by exactly 10)
// and the test flips to VERIFICATION SUCCESSFUL.
library Pay {
    function send(address payable to, uint256 val) internal {
        to.transfer(val);
    }
}

contract A {
    uint256 public snap_before;
    uint256 public snap_after;

    function run(address payable recipient) public {
        require(recipient != address(this));
        require(recipient != address(0));
        require(address(this).balance >= 10);
        snap_before = address(this).balance;
        Pay.send(recipient, 10);
        snap_after = address(this).balance;
        assert(snap_before - snap_after == 10);
    }
}

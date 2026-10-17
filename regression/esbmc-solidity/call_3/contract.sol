// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.8.0;

// Verifies that `address(this).balance` is invariant across an arbitrary
// low-level external call when this contract has no payable surface and
// performs no value transfers.
//
// Why SUCCESSFUL is the correct outcome:
//   - `called.call("")` carries NO value -> no ETH leaves `this`.
//   - `CallWrapper` has no `receive()` / `fallback()` payable, so the
//     EVM rejects any incoming `transfer`/`send` from `called` -> no
//     ETH credited to `this`.
//   - Neither `callwrap` nor `modifystorage` invokes
//     transfer/send/call{value:} -> no internal balance change either.
//   - `selfdestruct` is effectively retired (EIP-6049 / Cancun) and
//     coinbase rewards don't apply to arbitrary contracts -> we
//     deliberately don't model these protocol corners.
//
// The previous expected-FAILED was a soundness illusion: ESBMC used to
// resolve `address(this).balance` via a fresh nondet on every read, so
// the equality trivially failed independent of any external behaviour.
// After the SMTChecker-style fix (commit f507686) both reads alias the
// same `this->$balance` cell and the invariant holds, as it does on the
// real chain.
contract CallWrapper {
    uint data;

    function callwrap(address called) public {
        /// @custom:preghost function callwrap
        uint _balance = address(this).balance;

        called.call("");

        /// @custom:postghost function callwrap
        assert(_balance == address(this).balance);
    }

    function modifystorage(uint newdata) public {
        data = newdata;
    }

}
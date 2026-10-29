// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

// Regression for `model_transaction` revert semantics: when a contract is
// deployed via `new C{value:V}()` and the dynamic instance is stored in a
// state variable, subsequent `address(stateVar).balance` reads must reflect
// the actual heap allocation's $balance, not a NONDET that comes from the
// SMT solver picking the early-return path of the old (buggy) lowering.
//
// Pre-fix, model_transaction emitted `if (this.balance < val) return false;`
// which let the SMT pick the failure branch, exit createAndEndowD/Probe ctor
// early, and leave `Probe.v` at its initial NONDET value. The subsequent
// `address(v).balance` then resolved through SAME-OBJECT(...) checks that
// fell through to a fresh `invalid_object` symbol per read, so two reads of
// the same balance returned unrelated values.
//
// Post-fix, model_transaction emits `if (this.balance < val) __ESBMC_assume(false);`
// matching the established pattern in get_transfer_definition. The
// insufficient-balance path is pruned at the SMT level, the constructor
// runs to completion, `Probe.v` is properly bound to the heap allocation,
// and address(v).balance reads reflect the actual struct field.

contract Vault {
    constructor() payable {}
    function withdraw(address payable to, uint256 amt) external {
        to.transfer(amt);
    }
}

contract Probe {
    Vault v;
    constructor() payable { v = new Vault{value: 100}(); }
    function check() external {
        uint pre  = address(v).balance;
        v.withdraw(payable(address(0x1234)), 30);
        uint post = address(v).balance;
        assert(pre == post + 30);
    }
}

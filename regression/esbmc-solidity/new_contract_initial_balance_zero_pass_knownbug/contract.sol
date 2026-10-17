// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

// Audit finding F6: `new C()` (no `{value: V}`, non-payable ctor) must
// deploy with $balance = 0 per real EVM semantics. Today the model uses
// `_ndt_uint` for balance_init (solidity_convert_builtin.cpp:131-157),
// over-approximating to nondet. SMT picks any nondet value, so this
// assertion fails. Fix: replace `_ndt_uint` with `gen_zero` in the
// non-payable branch (S1.1).
contract Vault {}

contract H {
    Vault v;
    constructor() { v = new Vault(); }
    function check_initial_balance_zero() public view {
        assert(address(v).balance == 0);
    }
}

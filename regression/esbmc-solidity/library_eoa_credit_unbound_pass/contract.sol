// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

// Regression: `addr.balance` on an opaque address under `--unbound`
// now routes through the EOA balance map.  Previously the unbound
// path short-circuited to a fresh `nondet_uint()` before reaching
// `get_builtin_property_expr`, so credits written by
// transfer/send/call into `sol_eoa_balance_array` were invisible on
// subsequent reads.
//
// Fix (src/solidity-frontend/solidity_convert_expr.cpp): gate the
// short-circuit on `mem_name != "balance"`.  Balance now always
// calls `_ESBMC_eoa_balance_of`, whose `_ESBMC_eoa_get_or_init`
// helper still produces a nondet initial value for first-sight
// addresses — identical soundness to the old short-circuit — but
// honours credits afterwards.
contract C {
    function credit(address payable eoa) public {
        eoa.transfer(10);
    }

    function test(address payable eoa) public {
        require(eoa != address(0));
        require(eoa != address(this));
        require(eoa.balance == 100);
        credit(eoa);
        // After the credit, eoa.balance must be exactly 110.
        assert(eoa.balance == 110);
    }
}

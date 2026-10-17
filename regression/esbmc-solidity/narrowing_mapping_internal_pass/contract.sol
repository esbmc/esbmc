// SPDX-License-Identifier: MIT
// Regression for Block 1: mapping-key XOR-fold typecasts must be
// tagged internal and skipped by cast_overflow_check. Without the
// fix, `--unsigned-overflow-check` fires four spurious "Narrowing
// cast overflow on typecast" claims per mapping access (256→64 hash
// fold), which both masks real bugs AND prevents --multi-property
// --k-induction from converging. This contract has no real overflow
// and no user-written narrowing, so after the fix it MUST verify
// successfully under `--unsigned-overflow-check`.
pragma solidity >=0.8.0;

contract T {
    mapping(address => uint256) public balances;

    function set(address k, uint256 v) public {
        balances[k] = v;
    }
}

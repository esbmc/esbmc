// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

// B1 — EVM revert state-rollback semantics.  Exercises the new
// build_revert_rollback_block() lowering for `require(cond)` and
// `revert()`.  Before B1 these lowered to `__ESBMC_assume(false)` /
// `__ESBMC_assume(cond)`, pruning the path with the pre-revert state
// writes still in the SSA.  After B1, `*this` is restored to its
// function-entry snapshot before the function returns, matching real
// EVM semantics.
//
// This pass case demonstrates a classic guarded-mutation pattern:
// the deposit/withdraw checks must hold for any feasible balance
// update, and a successful path leaves the contract in a coherent
// state.  The harness exercises every dispatcher iteration with
// nondet inputs; with --bound, balances persist across calls.
contract Bank {
    mapping(address => uint) public balances;
    uint public totalDeposits;

    function deposit(uint amt) public {
        require(amt > 0, "amount must be positive");
        require(amt <= 1000, "deposit cap");
        balances[msg.sender] += amt;
        totalDeposits += amt;
    }

    function withdraw(uint amt) public {
        require(amt > 0, "amount must be positive");
        require(balances[msg.sender] >= amt, "insufficient");
        balances[msg.sender] -= amt;
        totalDeposits -= amt;
    }
}

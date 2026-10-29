// SPDX-License-Identifier: MIT
// Regression for the auto-CVC5 over-selection in src/esbmc/esbmc_parseoptions.cpp
// Pattern B (lines ~846-870 / 1215-1234). Pre-fix the heuristic forced
// --cvc5 whenever ALL of `--k-induction` + >=2 contracts + value-call
// (.transfer/.send/.call{value:}) held, regardless of whether the
// `this->$address` 256-bit-equality dispatcher chain was actually
// materialized. That chain exists only under bounded inter-contract
// modelling (--bound, or --reentry-check which programmatically enables
// --bound through the reentry harness); without either, ESBMC
// nondeterministically models external calls and the chain doesn't
// exist, so CVC5's array+datatype advantage evaporates and the default
// fallback (bitwuzla -> cvc5 -> boolector -> z3) is the right pick.
//
// This canary triggers Pattern B's three pre-fix conjuncts (2 contracts,
// .transfer, --k-induction) WITHOUT --bound and WITHOUT --reentry-check.
// Pre-fix the heuristic auto-selected CVC5; post-fix the new fourth
// conjunct `(cmdline.isset("bound") || cmdline.isset("reentry-check"))`
// gates the override off and the default fallback picks Bitwuzla. The
// `^Solving with solver Bitwuzla` regex line in test.desc anchors the
// post-fix solver-choice; both regexes must match for the CORE oracle
// to pass.
//
// Originally pinned as KNOWNBUG (commit 2c7e773cdc); flipped to CORE in
// the heuristic-narrowing fix at commit 478ed04fcf. The assertion
// `total == before_total - amount` is a true post-decrement invariant,
// so VERIFICATION SUCCESSFUL is the correct verdict regardless of
// solver.
pragma solidity >=0.8.0;
contract Vault {
    uint256 public total;

    function deposit(uint256 amount) public payable {
        total += amount;
    }

    function withdraw(address payable recipient, uint256 amount) public {
        require(total >= amount);
        uint256 before_total = total;
        total -= amount;
        recipient.transfer(amount);
        assert(total == before_total - amount);
    }
}

contract Sink {
    receive() external payable {}
}

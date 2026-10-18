// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

// T2.2 — positive regression coverage for try/catch state rollback
// (failing-transfer case). target's payOut pre-decrements bookkeeping
// then attempts a transfer; if transfer fails the call reverts and
// the pre-decrement is undone — catch should see ledger[u] == amt.
// Already covered by the if/else nondet-split lowering plus the
// transfer-fail `__ESBMC_assume(false)` (solidity_convert_call.cpp).
// The success arm is feasibility-pruned on transfer-fail; the catch
// arm sits in the else branch so SSA branching keeps the pre-decrement
// write out of catch's view.
contract Target {
    mapping(address => uint256) public ledger;

    constructor() {
        // Some baseline ledger entry so pre-decrement is meaningful.
    }

    function setBalance(address u, uint256 v) external {
        ledger[u] = v;
    }

    function payOut(address payable u, uint256 amt) external {
        require(ledger[u] >= amt);
        ledger[u] -= amt;
        u.transfer(amt);
    }
}

contract H {
    Target target;

    constructor() {
        target = new Target();
    }

    function check(address payable u, uint256 amt) external {
        require(amt != 0);
        target.setBalance(u, amt);  // ledger[u] = amt
        // amt is also Target's required transferable balance; if Target's
        // contract balance is zero, the transfer fails inside payOut.
        try target.payOut(u, amt) {
            // success path
        } catch {
            // catch should see ledger[u] == amt (pre-decrement undone).
            assert(target.ledger(u) == amt);
        }
    }
}

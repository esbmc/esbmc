// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

// B2 — Frame-context consistency.  Within a single function
// invocation (one call frame), every read of `msg.sender`,
// `msg.value`, `block.number`, `block.timestamp`, `tx.origin`,
// and `tx.gasprice` must return the same value as every other
// read of that same field in the same frame.  The dispatcher's
// `_sol_per_tx_reseed()` only fires at the *top* of each
// while-loop iteration; method bodies merely read the globals,
// they do not re-randomise them.  This test asserts that
// invariant directly: state is mutated between the two reads
// (so any code path that re-fired `_sol_per_tx_reseed` between
// them would break the assertion), but the two reads must agree.
contract C {
    uint public counter;

    function checkSender() public {
        address s1 = msg.sender;
        counter += 1;
        address s2 = msg.sender;
        assert(s1 == s2);
    }

    function checkValue() public payable {
        uint v1 = msg.value;
        counter += 1;
        uint v2 = msg.value;
        assert(v1 == v2);
    }

    function checkBlock() public {
        uint n1 = block.number;
        uint t1 = block.timestamp;
        counter += 1;
        uint n2 = block.number;
        uint t2 = block.timestamp;
        assert(n1 == n2);
        assert(t1 == t2);
    }

    function checkTx() public {
        address o1 = tx.origin;
        uint g1 = tx.gasprice;
        counter += 1;
        address o2 = tx.origin;
        uint g2 = tx.gasprice;
        assert(o1 == o2);
        assert(g1 == g2);
    }
}

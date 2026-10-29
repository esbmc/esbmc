// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

// A CONDITION'S TEMPORARIES MUST BE BUILT BEFORE THE CONDITION IS TESTED.
//
// Converting `b == bytes32(uint256(K))` materialises the right operand into an
// `_ESBMC_auxN` temporary, queued on the shared front-block pending list. That
// list has two readers, and they disagreed about who owns an entry:
// `flush_pending_into_body` takes a base index and deliberately leaves anything
// queued before the body alone, while `get_block` drained the list whole. For a
// BRACED body get_block ran first and won, so the goto program emitted
//
//     bytes_static_equal(&b, &_ESBMC_aux18)      <- reads it
//     DECL _ESBMC_aux18                          <- declares it
//     _ESBMC_aux18 = bytes_static_from_uint(1,32)<- builds it
//
// The comparison read an unconstrained struct, so `b == <const>` constrained
// `b` not at all. The brace-less spelling of the identical source was already
// correct because it never reached get_block -- the two syntactic positions of
// one construct disagreed, and only one of them was covered.
//
// THIS FILE IS USED BY THREE TESTS AND NEEDS ALL THREE. `contradiction` and
// `contradictionWhile` catch the defect (they must be SUCCESSFUL, and were
// FAILED before the fix); `reachable` is the vacuity guard (it must be FAILED).
// Without `reachable` a harness that stopped entering the unit at all would
// make the other two pass while proving nothing -- "SUCCESSFUL" is exactly the
// verdict an unreachable assertion produces.
contract C
{
  uint256 public tag;

  // VACUITY GUARD. `b` CAN be bytes32(1), the braced body IS reachable, and
  // the assertion therefore MUST be violated. If this ever reports SUCCESSFUL,
  // the other two tests in this family have stopped testing anything.
  function reachable(bytes32 b) external
  {
    if (b == bytes32(uint256(1)))
    {
      tag = 1;
    }
    assert(tag != 1);
  }

  // DEFECT GUARD, if-statement. One bytes32 cannot equal two different
  // constants, so the body is unreachable and the assertion holds.
  function contradiction(bytes32 b) external
  {
    if (b == bytes32(uint256(1)) && b == bytes32(uint256(2)))
    {
      tag = 2;
    }
    assert(tag != 2);
  }

  // DEFECT GUARD, while-statement. Same contradiction behind a braced loop
  // body, which reaches get_block by a different route (get_statement's
  // WhileStatement arm calls get_block directly, with no base snapshot of its
  // own). Measured FAILED before the fix, exactly like the if form.
  function contradictionWhile(bytes32 b) external
  {
    while (b == bytes32(uint256(1)) && b == bytes32(uint256(2)))
    {
      tag = 3;
      break;
    }
    assert(tag != 3);
  }
}

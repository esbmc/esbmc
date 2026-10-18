// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

// KNOWNBUG: a `for` condition's operand temporaries are still built after the
// comparison that reads them.
//
// The same defect as solidity_braced_body_cond_aux_{if,while}: converting
// `b == bytes32(uint256(K))` queues the temporary holding the constant on the
// shared front block, and `get_block` drains that queue into the loop BODY --
// which, for a `for`, executes after the condition is first tested. So the
// comparison reads an unconstrained struct and constrains `b` not at all.
//
// The fix for `if` and `while` (`hoist_operands_read_by`, lifting exactly the
// pending statements whose declared symbol the condition references) is NOT
// wired into the ForStatement arm. Deliberately: `while` already had to choose
// "build the operands once, before the loop", and that choice is wrong for a
// condition operand depending on state the body mutates. Extending it to `for`
// without settling that would spread one half-answer to another place, so the
// gap is pinned here instead of half-closed.
//
// MEASURED, and not a bound artefact -- reproduced at the default unwind and at
// --unwind 4 and --unwind 8. The paired control is `do-while`, which is NOT
// affected: its body executes before its condition, so draining into the body
// lands the temporary before the use. That control is what makes this a result
// rather than a guess -- without it, a "SUCCESSFUL" here could equally have
// been a loop bound cutting the second iteration.
//
// This asserts a Solidity tautology: one bytes32 cannot equal two different
// constants, so the body is unreachable and `n` must stay 0.
contract C
{
  function forCond(bytes32 b) public pure
  {
    uint256 n = 0;
    for (; b == bytes32(uint256(1)) && b == bytes32(uint256(2)); )
    {
      n++;
      break;
    }
    assert(n == 0);
  }
}

#pragma once

#include <goto-programs/goto_functions.h>
#include <llm/llm_client.h>
#include <util/config/options.h>
#include <util/symtab/context.h>

/// --llm-invariants: ask a language model for loop invariants with
/// chain-of-thought prompting, as Pirzada et al. do ("LLM-Generated Invariants
/// for Bounded Model Checking Without Loop Unrolling", ASE 2024), and refine
/// them counterexample-guided, which that work does not.
///
/// Each round renders the program for @p client, parses the invariants it
/// answers, and checks them with the loop-invariant havoc schema in a quiet
/// BMC run on a copy of the program. Conjuncts whose base case or inductive
/// step fails are dropped until the rest are inductive (Houdini); the failures
/// and any assertion they still leave unproved go back to the model for the
/// next round, at most @p max_rounds rounds. The survivors are attached as
/// synthesised LOOP_INVARIANT markers, which the caller's goto_loop_invariant
/// checks again, so a wrong answer can never fake a proof. A conjunct the
/// check could not vouch for is dropped, since one that fails in the caller's
/// run makes it report FAILED.
void llm_synthesise_loop_invariants(
  goto_functionst &goto_functions,
  contextt &context,
  const optionst &options,
  llm::clientt &client,
  unsigned max_rounds,
  bool use_frame_rule);

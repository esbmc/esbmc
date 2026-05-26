/// \file
/// Interval Analysis

#ifndef CPROVER_ANALYSES_INTERVAL_ANALYSIS_H
#define CPROVER_ANALYSES_INTERVAL_ANALYSIS_H

#include <goto-programs/goto_functions.h>

// Where to add the assumes?
enum class INTERVAL_INSTRUMENTATION_MODE
{
  NO_INSTRUMENTATION,
  ALL_INSTRUCTIONS_FULL, // All instructions with all symbols belonging to the function
  ALL_INSTRUCTIONS_LOCAL, // All instructions with the symbols affecting the instructions
  GUARD_INSTRUCTIONS_FULL, // Assume, Asserts, GOTO with all symbols belonging to the function
  GUARD_INSTRUCTIONS_LOCAL, // Assume, Asserts, GOTO with the symbols affecting the instruction.
  LOOP_MODE // Adds an assumption before, during and at the end of a loop with all symbols that affect it
};

void interval_analysis(
  goto_functionst &goto_functions,
  const namespacet &ns,
  const optionst &options,
  const INTERVAL_INSTRUMENTATION_MODE instrument_mode =
    INTERVAL_INSTRUMENTATION_MODE::GUARD_INSTRUCTIONS_LOCAL);

/// Post-k-induction loop-bounds instrumentation.
///
/// Re-runs interval analysis with `inductive_step_instruction` instructions
/// (k-induction's nondet havocs and entry-condition assumes) treated as
/// transparent no-ops, so the fixpoint at each loop head reflects the
/// *original* program's dataflow. For every loop, inserts a single
/// ASSUME(bounds) immediately before the loop's exit-test (the IF that
/// k-induction's preamble runs into) — i.e. between the havoc'd assignments
/// and the IF, so the bounds tighten the havoc'd values for the inductive
/// step. The inserted ASSUME is marked `inductive_step_instruction = true`
/// so base case and forward condition skip it.
///
/// Must run AFTER goto_k_induction(). Always runs the full ait fixpoint
/// and per-loop scan; inserts no ASSUME for loops where every variable's
/// bound simplifies to true (i.e. no useful tightening).
void instrument_loop_bounds_after_kind(
  goto_functionst &goto_functions,
  const namespacet &ns,
  const optionst &options);

#endif // CPROVER_ANALYSES_INTERVAL_ANALYSIS_H

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
    INTERVAL_INSTRUMENTATION_MODE::LOOP_MODE);

#endif // CPROVER_ANALYSES_INTERVAL_ANALYSIS_H

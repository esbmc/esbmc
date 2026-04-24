/// \file
/// Linear Equality Analysis

#pragma once

#include <goto-programs/goto_functions.h>
#include <util/options.h>

/**
 * @brief Run the linear equality analysis over goto_functions and instrument
 *        each loop with ASSUME(invariant) at the loop body entry.
 *
 * @param goto_functions  GOTO program to analyse and instrument (modified in-place).
 * @param ns              Namespace for type/symbol resolution.
 * @param options         Command-line options; recognises:
 *                          --linear-equality-analysis
 *                          --linear-equality-analysis-dump
 */
void linear_equality_analysis(
  goto_functionst &goto_functions,
  const namespacet &ns,
  const optionst &options);

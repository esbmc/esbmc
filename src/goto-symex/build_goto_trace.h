/*******************************************************************\

Module: Traces of GOTO Programs

Author: Daniel Kroening

Date: July 2005

\*******************************************************************/

#ifndef CPROVER_GOTO_SYMEX_BUILD_GOTO_TRACE_H
#define CPROVER_GOTO_SYMEX_BUILD_GOTO_TRACE_H

#include <goto-symex/goto_symex_state.h>
#include <goto-symex/goto_trace.h>
#include <goto-symex/symex_target_equation.h>

void build_goto_trace(
  const symex_target_equationt &target,
  smt_convt &smt_conv,
  goto_tracet &goto_trace);

void build_successful_goto_trace(
  const symex_target_equationt &target,
  const namespacet &ns,
  goto_tracet &goto_trace);

bool is_valid_correctness_SSA_step(
  const namespacet & ns,
  symex_target_equationt::SSA_stepst::const_iterator & step);

#endif

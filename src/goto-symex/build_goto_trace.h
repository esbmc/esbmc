#ifndef CPROVER_GOTO_SYMEX_BUILD_GOTO_TRACE_H
#define CPROVER_GOTO_SYMEX_BUILD_GOTO_TRACE_H

#include <goto-symex/goto_symex_state.h>
#include <goto-symex/goto_trace.h>
#include <goto-symex/symex_target_equation.h>

void build_goto_trace(
  const symex_target_equationt &target,
  smt_convt &smt_conv,
  goto_tracet &goto_trace,
  const bool &is_compact_trace);

void build_successful_goto_trace(
  const symex_target_equationt &target,
  const namespacet &ns,
  goto_tracet &goto_trace);

expr2tc build_lhs(smt_convt &smt_conv, const expr2tc &lhs);
expr2tc build_rhs(smt_convt &smt_conv, const expr2tc &rhs);

#endif

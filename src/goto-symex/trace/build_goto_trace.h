#ifndef CPROVER_GOTO_SYMEX_BUILD_GOTO_TRACE_H
#define CPROVER_GOTO_SYMEX_BUILD_GOTO_TRACE_H

#include <goto-symex/trace/goto_trace.h>
#include <goto-symex/equation/symex_target_equation.h>

/// `keep_hidden_steps` retains the steps flagged hidden, which a trace
/// otherwise drops. Only for consumers that harvest values from the trace
/// instead of printing it: hidden steps carry frontend-synthesised state.
void build_goto_trace(
  const symex_target_equationt &target,
  smt_convt &smt_conv,
  goto_tracet &goto_trace,
  bool keep_hidden_steps = false);

void build_successful_goto_trace(
  const symex_target_equationt &target,
  const namespacet &ns,
  goto_tracet &goto_trace);

expr2tc build_lhs(smt_convt &smt_conv, const expr2tc &lhs);
expr2tc build_rhs(smt_convt &smt_conv, const expr2tc &rhs);

#endif

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
  const boost::shared_ptr<symex_target_equationt> &target,
  boost::shared_ptr<smt_convt> &smt_conv,
  goto_tracet &goto_trace);

void build_successful_goto_trace(
  const boost::shared_ptr<symex_target_equationt> &target,
  const namespacet &ns,
  goto_tracet &goto_trace);

#endif

/*
 * goto_unwind.cpp
 *
 *  Created on: Jun 3, 2015
 *      Author: mramalho
 */

#include <util/std_expr.h>
#include <util/expr_util.h>

#include "goto_k_induction.h"
#include "remove_skip.h"

bool check_var_name(const exprt &expr)
{
  std::size_t found = expr.identifier().as_string().find("__ESBMC_");
  if(found != std::string::npos)
    return false;

  found = expr.identifier().as_string().find("__CPROVER");
  if(found != std::string::npos)
    return false;

  found = expr.identifier().as_string().find("return_value___");
  if(found != std::string::npos)
    return false;

  if(expr.identifier().as_string() == "c::__func__"
     || expr.identifier().as_string() == "c::__PRETTY_FUNCTION__"
     || expr.identifier().as_string() == "c::__LINE__"
     || expr.identifier().as_string() == "c::pthread_lib::num_total_threads"
     || expr.identifier().as_string() == "c::pthread_lib::num_threads_running")
    return false;

  if(expr.location().file().as_string() == "<built-in>"
     || expr.cmt_location().file().as_string() == "<built-in>"
     || expr.type().location().file().as_string() == "<built-in>"
     || expr.type().cmt_location().file().as_string() == "<built-in>")
    return false;

  return true;
}

void goto_k_induction(
  goto_functionst& goto_functions,
  const namespacet &ns,
  message_handlert& message_handler)
{
  Forall_goto_functions(it, goto_functions)
    if(it->second.body_available)
      goto_k_inductiont(it->second, ns, message_handler);

  goto_functions.update();
}

void goto_k_inductiont::goto_k_induction()
{
}

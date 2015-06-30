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

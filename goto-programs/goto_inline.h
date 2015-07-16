/*******************************************************************\

Module: Function Inlining

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#ifndef CPROVER_GOTO_INLINE_H
#define CPROVER_GOTO_INLINE_H

#include <std_types.h>
#include <hash_cont.h>

#include <message_stream.h>

#include "goto_functions.h"

// do a full inlining
void goto_inline(
  goto_functionst &goto_functions,
  optionst &options,
  const namespacet &ns,
  goto_programt &dest,
  message_handlert &message_handler);

void goto_inline(
  goto_functionst &goto_functions,
  optionst &options,
  const namespacet &ns,
  message_handlert &message_handler);

// inline those functions marked as "inlined"
// and functions with less than _smallfunc_limit instructions
void goto_partial_inline(
  goto_functionst &goto_functions,
  optionst &options,
  const namespacet &ns,
  message_handlert &message_handler,
  unsigned _smallfunc_limit = 0);

class goto_inlinet:public message_streamt
{
public:
  goto_inlinet(
    goto_functionst &_goto_functions,
    optionst &_options,
    const namespacet &_ns,
    message_handlert &_message_handler):
    message_streamt(_message_handler),
    smallfunc_limit(0),
    goto_functions(_goto_functions),
    options(_options),
    ns(_ns)
  {
  }

  void goto_inline(goto_programt &dest);
  void goto_inline_rec(goto_programt &dest, bool full);

  // inline single instruction at 'target'
  // returns true if any inlining was performed
  // set 'full' to perform this recursively
  bool inline_instruction(
    goto_programt &dest,
    bool full,
    goto_programt::targett &target);

  unsigned smallfunc_limit;

protected:
  goto_functionst &goto_functions;
  optionst &options;
  const namespacet &ns;

  void expand_function_call(
    goto_programt &dest,
    goto_programt::targett &target,
    const exprt &lhs,
    const exprt &function,
    const exprt::operandst &arguments,
    const exprt &constrain,
    bool recursive);

  void replace_return(
    goto_programt &body,
    const exprt &lhs,
    const exprt &constrain);

  void parameter_assignments(
    const locationt &location,
    const code_typet &code_type,
    const exprt::operandst &arguments,
    goto_programt &dest);

  typedef hash_set_cont<irep_idt, irep_id_hash> recursion_sett;
  recursion_sett recursion_set;

  typedef hash_set_cont<irep_idt, irep_id_hash> no_body_sett;
  no_body_sett no_body_set;

public:
  // Set of function names that have been inlined into the function we're
  // dealing with right now. Fairly hacky, could be improved.
  std::set<std::string> inlined_funcs;
};

#endif

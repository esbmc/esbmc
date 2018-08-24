/*******************************************************************\

Module: Abstract Interpretation

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

/// \file
/// Abstract Interpretation

#include "ai.h"

#include <cassert>
#include <memory>
#include <sstream>

#include <util/std_code.h>
#include <util/std_expr.h>

void ai_baset::output(const goto_functionst &goto_functions, std::ostream &out)
  const
{
  forall_goto_functions(f_it, goto_functions)
  {
    if(f_it->second.body_available)
    {
      out << "////\n";
      out << "//// Function: " << f_it->first << "\n";
      out << "////\n";
      out << "\n";

      forall_goto_program_instructions(i_it, f_it->second.body)
      {
        out << "**** " << i_it->location_number << " " << i_it->location
            << "\n";

        abstract_state_before(i_it)->output(out);
        out << "\n";
        i_it->dump();
        out << "\n";
      }
    }
  }
}

void ai_baset::entry_state(const goto_functionst &goto_functions)
{
  // find the 'entry function'

  goto_functionst::function_mapt::const_iterator f_it =
    goto_functions.function_map.find(goto_functions.main_id());

  if(f_it != goto_functions.function_map.end())
    entry_state(f_it->second.body);
}

void ai_baset::entry_state(const goto_programt &goto_program)
{
  // The first instruction of 'goto_program' is the entry point
  get_state(goto_program.instructions.begin()).make_entry();
}

void ai_baset::initialize(const goto_functiont &goto_function)
{
  initialize(goto_function.body);
}

void ai_baset::initialize(const goto_programt &goto_program)
{
  // we mark everything as unreachable as starting point

  forall_goto_program_instructions(i_it, goto_program)
    get_state(i_it).make_bottom();
}

void ai_baset::initialize(const goto_functionst &goto_functions)
{
  forall_goto_functions(it, goto_functions)
    initialize(it->second);
}

void ai_baset::finalize()
{
  // Nothing to do per default
}

goto_programt::const_targett ai_baset::get_next(working_sett &working_set)
{
  assert(!working_set.empty());

  working_sett::iterator i = working_set.begin();
  goto_programt::const_targett l = i->second;
  working_set.erase(i);

  return l;
}

bool ai_baset::fixedpoint(
  const goto_programt &goto_program,
  const goto_functionst &goto_functions,
  const namespacet &ns)
{
  working_sett working_set;

  // Put the first location in the working set
  if(!goto_program.empty())
    put_in_working_set(working_set, goto_program.instructions.begin());

  bool new_data = false;

  while(!working_set.empty())
  {
    goto_programt::const_targett l = get_next(working_set);

    // goto_program is really only needed for iterator manipulation
    if(visit(l, working_set, goto_program, goto_functions, ns))
      new_data = true;
  }

  return new_data;
}

bool ai_baset::visit(
  goto_programt::const_targett l,
  working_sett &working_set,
  const goto_programt &goto_program,
  const goto_functionst &goto_functions,
  const namespacet &ns)
{
  bool new_data = false;

  statet &current = get_state(l);

  goto_programt::const_targetst successors;
  goto_program.get_successors(l, successors);

  for(const auto &to_l : successors)
  {
    if(to_l == goto_program.instructions.end())
      continue;

    std::unique_ptr<statet> tmp_state(make_temporary_state(current));

    statet &new_values = *tmp_state;

    bool have_new_values = false;

    if(l->is_function_call() && !goto_functions.function_map.empty())
    {
      // this is a big special case
      const code_function_call2t &code = to_code_function_call2t(l->code);

      if(do_function_call_rec(l, to_l, code.function, goto_functions, ns))
        have_new_values = true;
    }
    else
    {
      // initialize state, if necessary
      get_state(to_l);

      new_values.transform(l, to_l, *this, ns);

      if(merge(new_values, l, to_l))
        have_new_values = true;
    }

    if(have_new_values)
    {
      new_data = true;
      put_in_working_set(working_set, to_l);
    }
  }

  return new_data;
}

bool ai_baset::do_function_call(
  goto_programt::const_targett l_call,
  goto_programt::const_targett l_return,
  const goto_functionst &goto_functions,
  const goto_functionst::function_mapt::const_iterator f_it,
  const namespacet &ns)
{
  // initialize state, if necessary
  get_state(l_return);

  assert(l_call->is_function_call());

  const goto_functiont &goto_function = f_it->second;

  if(!goto_function.body_available)
  {
    // if we don't have a body, we just do an edige call -> return
    std::unique_ptr<statet> tmp_state(make_temporary_state(get_state(l_call)));
    tmp_state->transform(l_call, l_return, *this, ns);

    return merge(*tmp_state, l_call, l_return);
  }

  assert(!goto_function.body.instructions.empty());

  // This is the edge from call site to function head.

  {
    // get the state at the beginning of the function
    goto_programt::const_targett l_begin =
      goto_function.body.instructions.begin();
    // initialize state, if necessary
    get_state(l_begin);

    // do the edge from the call site to the beginning of the function
    std::unique_ptr<statet> tmp_state(make_temporary_state(get_state(l_call)));
    tmp_state->transform(l_call, l_begin, *this, ns);

    bool new_data = false;

    // merge the new stuff
    if(merge(*tmp_state, l_call, l_begin))
      new_data = true;

    // do we need to do/re-do the fixedpoint of the body?
    if(new_data)
      fixedpoint(goto_function.body, goto_functions, ns);
  }

  // This is the edge from function end to return site.

  {
    // get location at end of the procedure we have called
    goto_programt::const_targett l_end =
      --goto_function.body.instructions.end();
    assert(l_end->is_end_function());

    // do edge from end of function to instruction after call
    const statet &end_state = get_state(l_end);

    if(end_state.is_bottom())
      return false; // function exit point not reachable

    std::unique_ptr<statet> tmp_state(make_temporary_state(end_state));
    tmp_state->transform(l_end, l_return, *this, ns);

    // Propagate those
    return merge(*tmp_state, l_end, l_return);
  }
}

bool ai_baset::do_function_call_rec(
  goto_programt::const_targett l_call,
  goto_programt::const_targett l_return,
  const expr2tc &function,
  const goto_functionst &goto_functions,
  const namespacet &ns)
{
  assert(!goto_functions.function_map.empty());
  bool new_data = false;

  // This is quite a strong assumption on the well-formedness of the program.
  // It means function pointers must be removed before use.
  if(is_symbol2t(function))
  {
    const irep_idt &identifier = to_symbol2t(function).thename;

    goto_functionst::function_mapt::const_iterator it =
      goto_functions.function_map.find(identifier);

    assert(it != goto_functions.function_map.end());

    new_data = do_function_call(l_call, l_return, goto_functions, it, ns);
  }

  return new_data;
}

void ai_baset::sequential_fixedpoint(
  const goto_functionst &goto_functions,
  const namespacet &ns)
{
  goto_functionst::function_mapt::const_iterator f_it =
    goto_functions.function_map.find(goto_functions.main_id());

  if(f_it != goto_functions.function_map.end())
    fixedpoint(f_it->second.body, goto_functions, ns);
}

/*******************************************************************\

Module: Value Set Propagation

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include <cassert>
#include <goto-programs/static_analysis.h>
#include <memory>
#include <util/expr_util.h>
#include <util/std_code.h>
#include <util/std_expr.h>

expr2tc abstract_domain_baset::get_guard(
  locationt from,
  locationt to) const
{
  if(!from->is_goto())
    return gen_true_expr();

  locationt next=from;
  next++;

  if(next==to)
  {
    expr2tc tmp = not2tc(from->guard);
    return tmp;
  }

  return from->guard;
}

expr2tc abstract_domain_baset::get_return_lhs(locationt to) const
{
  // get predecessor of "to"

  to--;

  if(to->is_end_function())
    return expr2tc();

  // must be the function call
  assert(to->is_function_call());

  const code_function_call2t &code = to_code_function_call2t(to->code);

  return code.ret;
}

void static_analysis_baset::operator()(
  const goto_functionst &goto_functions)
{
  initialize(goto_functions);
  fixedpoint(goto_functions);
}

void static_analysis_baset::operator()(
  const goto_programt &goto_program)
{
  initialize(goto_program);
  goto_functionst goto_functions;
  fixedpoint(goto_program, goto_functions);
}

void static_analysis_baset::output(
  const goto_functionst &goto_functions,
  std::ostream &out) const
{
  for(const auto & f_it : goto_functions.function_map)
  {
    out << "////" << std::endl;
    out << "//// Function: " << f_it.first << std::endl;
    out << "////" << std::endl;
    out << std::endl;

    output(f_it.second.body, f_it.first, out);
  }
}

void static_analysis_baset::output(
  const goto_programt &goto_program,
  const irep_idt &identifier,
  std::ostream &out) const
{
  forall_goto_program_instructions(i_it, goto_program)
  {
    out << "**** " << i_it->location << std::endl;

    get_state(i_it).output(ns, out);
    out << std::endl;
    i_it->output_instruction(ns, identifier, out);
    out << std::endl;
  }
}

void static_analysis_baset::generate_states(
  const goto_functionst &goto_functions)
{
  for(const auto & f_it : goto_functions.function_map)
    generate_states(f_it.second.body);
}

void static_analysis_baset::generate_states(
  const goto_programt &goto_program)
{
  forall_goto_program_instructions(i_it, goto_program)
    generate_state(i_it);
}

void static_analysis_baset::update(
  const goto_functionst &goto_functions)
{
  for(const auto & f_it : goto_functions.function_map)
    update(f_it.second.body);
}

void static_analysis_baset::update(
  const goto_programt &goto_program)
{
  locationt previous;
  bool first=true;

  forall_goto_program_instructions(i_it, goto_program)
  {
    // do we have it already?
    if(!has_location(i_it))
    {
      generate_state(i_it);

      if(!first)
        merge(get_state(i_it), get_state(previous));
    }

    first=false;
    previous=i_it;
  }
}

static_analysis_baset::locationt static_analysis_baset::get_next(
  working_sett &working_set)
{
  assert(!working_set.empty());

  working_sett::iterator i=working_set.begin();
  locationt l=i->second;
  working_set.erase(i);

  return l;
}

bool static_analysis_baset::fixedpoint(
  const goto_programt &goto_program,
  const goto_functionst &goto_functions)
{
  if(goto_program.instructions.empty())
    return false;

  working_sett working_set;

  put_in_working_set(
    working_set,
    goto_program.instructions.begin());

  bool new_data=false;

  while(!working_set.empty())
  {
    locationt l=get_next(working_set);

    if(visit(l, working_set, goto_program, goto_functions))
      new_data=true;
  }

  return new_data;
}

bool static_analysis_baset::visit(
  locationt l,
  working_sett &working_set,
  const goto_programt &goto_program,
  const goto_functionst &goto_functions)
{
  bool new_data=false;

  statet &current=get_state(l);

  current.seen=true;

  goto_programt::const_targetst successors;

  goto_program.get_successors(l, successors);

  for(goto_programt::const_targetst::const_iterator
      it=successors.begin();
      it!=successors.end();
      it++)
  {
    locationt to_l=*it;

    if(to_l==goto_program.instructions.end())
      continue;

    std::unique_ptr<statet> tmp_state(make_temporary_state(current));

    statet &new_values=*tmp_state;

    // Do we want to pull new variables into the new state when tracking?
    bool merge_new_vals = true;

    if(l->is_function_call())
    {
      // this is a big special case
      const code_function_call2t &code = to_code_function_call2t(l->code);

      do_function_call_rec(
        l,
        code.function,
        code.operands,
        new_values,
        goto_functions);

      // Don't track variables that are new in the callee into the callers
      // state. We don't want its arguments and local variables cluttering
      // our own value set.
      merge_new_vals = false;
    }
    else
      new_values.transform(ns, l, to_l);

    statet &other=get_state(to_l);

    bool have_new_values=
      merge(other, new_values, merge_new_vals);

    if(have_new_values)
      new_data=true;

    if(have_new_values || !other.seen)
      put_in_working_set(working_set, to_l);
  }

  return new_data;
}

void static_analysis_baset::do_function_call(
  locationt l_call,
  const goto_functionst &goto_functions,
  const goto_functionst::function_mapt::const_iterator f_it,
  const std::vector<expr2tc> &arguments __attribute__((unused)), // XXX -- why?
  statet &new_state)
{
  const goto_functiont &goto_function=f_it->second;

  if(!goto_function.body_available)
    return; // do nothing

  assert(!goto_function.body.instructions.empty());

  {
    // get the state at the beginning of the function
    locationt l_begin=goto_function.body.instructions.begin();

    // do the edge from the call site to the beginning of the function
    new_state.transform(ns, l_call, l_begin);

    statet &begin_state=get_state(l_begin);

    bool new_data=false;

    // merge the new stuff. Place local state of callers into the value set of
    // the callee, it might end up accessing a pointer to the callers local
    // variables.
    if(merge(begin_state, new_state, true))
      new_data=true;

    // do each function at least once
    if(functions_done.find(f_it->first)==
       functions_done.end())
    {
      new_data=true;
      functions_done.insert(f_it->first);
    }

    // do we need to do the fixedpoint of the body?
    if(new_data)
    {
      // recursive call
      fixedpoint(goto_function.body, goto_functions);
    }
  }

  {
    // get location at end of procedure
    locationt l_end=--goto_function.body.instructions.end();

    assert(l_end->is_end_function());

    statet &end_of_function=get_state(l_end);

    // do edge from end of function to instruction after call
    locationt l_next=l_call;
    l_next++;
    end_of_function.transform(ns, l_end, l_next);

    // propagate those -- not exceedingly precise, this is,
    // as still it contains all the state from the
    // call site
    merge(new_state, end_of_function);
  }
}

void static_analysis_baset::do_function_call_rec(
  locationt l_call,
  const expr2tc &function,
  const std::vector<expr2tc> &arguments,
  statet &new_state,
  const goto_functionst &goto_functions)
{
  if (is_symbol2t(function))
  {
    irep_idt identifier = to_symbol2t(function).get_symbol_name();

    if(recursion_set.find(identifier)!=recursion_set.end())
    {
      // recursion detected!
      return;
    }
    else
      recursion_set.insert(identifier);

    goto_functionst::function_mapt::const_iterator it=
      goto_functions.function_map.find(identifier);

    if(it==goto_functions.function_map.end())
      throw "failed to find function "+id2string(identifier);

    do_function_call(
      l_call,
      goto_functions,
      it,
      arguments,
      new_state);

    recursion_set.erase(identifier);
  }
  else if (is_if2t(function))
  {
    const if2t ifval = to_if2t(function);
    std::unique_ptr<statet> n2(make_temporary_state(new_state));

    do_function_call_rec(
      l_call,
      ifval.true_value,
      arguments,
      new_state,
      goto_functions);

    do_function_call_rec(
      l_call,
      ifval.false_value,
      arguments,
      *n2,
      goto_functions);

    merge(new_state, *n2);
  }
  else if (is_dereference2t(function))
  {
    // get value set
    std::list<expr2tc> values;
    get_reference_set(l_call, function, values);

    std::unique_ptr<statet> state_from(make_temporary_state(new_state));

    // now call all of these
    for(std::list<expr2tc>::const_iterator it=values.begin();
        it!=values.end(); it++)
    {
      if (is_object_descriptor2t(*it))
      {
        const object_descriptor2t &obj = to_object_descriptor2t(*it);
        std::unique_ptr<statet> n2(make_temporary_state(new_state));
        do_function_call_rec(l_call, obj.object, arguments, *n2, goto_functions);
        merge(new_state, *n2);
      }
    }
  }
  else if (is_null_object2t(function))
  {
    // ignore, can't be a function
  }
  else if (is_member2t(function) || is_index2t(function))
  {
    // ignore, can't be a function
  }
  else
  {
    std::cerr << "unexpected function_call argument: "
              << get_expr_id(function) << std::endl;
    abort();
  }
}

void static_analysis_baset::fixedpoint(
  const goto_functionst &goto_functions)
{
  // do each function at least once

  for(goto_functionst::function_mapt::const_iterator
      it=goto_functions.function_map.begin();
      it!=goto_functions.function_map.end();
      it++)
    if(functions_done.find(it->first)==
       functions_done.end())
    {
      fixedpoint(it, goto_functions);
    }
}

bool static_analysis_baset::fixedpoint(
  const goto_functionst::function_mapt::const_iterator it,
  const goto_functionst &goto_functions)
{
  functions_done.insert(it->first);
  return fixedpoint(it->second.body, goto_functions);
}

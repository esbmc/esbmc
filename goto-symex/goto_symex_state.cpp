/*******************************************************************\

Module: Symbolic Execution

Author: Daniel Kroening, kroening@kroening.com
		Lucas Cordeiro, lcc08r@ecs.soton.ac.uk

\*******************************************************************/

#include <irep2.h>
#include <migrate.h>

#include <assert.h>
#include <global.h>
#include <malloc.h>
#include <map>
#include <sstream>

#include <i2string.h>
#include "../util/expr_util.h"

#include "reachability_tree.h"
#include "execution_state.h"
#include "goto_symex_state.h"
#include "goto_symex.h"
#include "crypto_hash.h"

goto_symex_statet::goto_symex_statet(renaming::level2t &l2, value_sett &vs,
                                     const namespacet &_ns)
    : guard(), level2(l2), value_set(vs), ns(_ns)
{
  use_value_set = true;
  depth = 0;
  thread_ended = false;
  guard.make_true();
}

goto_symex_statet::goto_symex_statet(const goto_symex_statet &state,
                                     renaming::level2t &l2,
                                     value_sett &vs)
  : level2(l2), value_set(vs), ns(state.ns)
{
  *this = state;
}

goto_symex_statet &
goto_symex_statet::operator=(const goto_symex_statet &state)
{
  depth = state.depth;
  thread_ended = state.thread_ended;
  guard = state.guard;
  source = state.source;
  function_frame = state.function_frame;
  unwind_map = state.unwind_map;
  function_unwind = state.function_unwind;
  use_value_set = state.use_value_set;
  call_stack = state.call_stack;
  return *this;
}

void goto_symex_statet::initialize(const goto_programt::const_targett & start, const goto_programt::const_targett & end, const goto_programt *prog, unsigned int thread_id)
{
  new_frame(thread_id);

  source.is_set=true;
  source.thread_nr = thread_id;
  source.pc=start;
  source.prog = prog;
  top().end_of_function=end;
  top().calling_location=symex_targett::sourcet(top().end_of_function, prog);
}

bool goto_symex_statet::constant_propagation(const exprt &expr) const
{
  static unsigned int with_counter=0;
  if(expr.is_constant()) return true;

  if(expr.id()==exprt::addrof)
  {
    if(expr.operands().size()!=1)
      throw "address_of expects one operand";

    return constant_propagation_reference(expr.op0());
  }
  else if(expr.id()==exprt::typecast)
  {
    if(expr.operands().size()!=1)
      throw "typecast expects one operand";

    return constant_propagation(expr.op0());
  }
  else if(expr.id()==exprt::plus)
  {
    forall_operands(it, expr)
      if(!constant_propagation(*it))
        return false;

    return true;
  }
#if 1
  else if(expr.id()==exprt::arrayof)
  {
    if(expr.operands().size()==1)
      if (expr.op0().id()==exprt::constant && expr.op0().type().id()!=typet::t_bool)
        return true;
  }
#endif
#if 1
  else if(expr.id()==exprt::with)
  {
	with_counter++;

	if (with_counter>6)
	{
		with_counter=0;
		return false;
	}

    //forall_operands(it, expr)
    //{
      if(!constant_propagation(expr.op0()))
      {
    	with_counter=0;
        return false;
      }
    //}
    with_counter=0;
    return true;
  }
#endif
  else if(expr.id()=="struct")
  {
    forall_operands(it, expr)
      if(!constant_propagation(*it))
        return false;

    return true;
  }

  else if(expr.id()=="union")
  {
    if(expr.operands().size()==1)
      return constant_propagation(expr.op0());
  }

  /* No difference
  else if(expr.id()==exprt::equality)
  {
    if(expr.operands().size()!=2)
	  throw "equality expects two operands";

    return (constant_propagation(expr.op0()) ||
           constant_propagation(expr.op1()));

  }
  */

  return false;
}

bool goto_symex_statet::constant_propagation_reference(const exprt &expr) const
{
  if(expr.id()==exprt::symbol)
    return true;
  else if(expr.id()==exprt::index)
  {
    if(expr.operands().size()!=2)
      throw "index expects two operands";

    return constant_propagation_reference(expr.op0()) &&
           constant_propagation(expr.op1());
  }
  else if(expr.id()==exprt::member)
  {
    if(expr.operands().size()!=1)
      throw "member expects one operand";

    return constant_propagation_reference(expr.op0());
  }
#if 1
  else if(expr.id()=="string-constant")
    return true;
#endif

  return false;
}

void goto_symex_statet::assignment(
  exprt &lhs,
  const exprt &rhs,
  bool record_value)
{
  crypto_hash hash;
  assert(lhs.id()=="symbol");
  assert(lhs.id()==exprt::symbol);

  const irep_idt &identifier= lhs.identifier();

  // identifier should be l0 or l1, make sure it's l1

  const std::string l1_identifier=top().level1.get_ident_name(identifier);

  exprt const_value;
  if(record_value && constant_propagation(rhs))
    const_value = rhs;
  else
    const_value.make_nil();

  irep_idt new_name = level2.make_assignment(l1_identifier, const_value, rhs);
  lhs.identifier(new_name);

  if(use_value_set)
  {
    // update value sets
    expr2tc l1_rhs;
    migrate_expr(rhs, l1_rhs);
    level2.get_original_name(l1_rhs);

    type2tc lhs_type;
    migrate_type(lhs.type(), lhs_type);

    expr2tc l1_lhs = expr2tc(new symbol2t(lhs_type, l1_identifier));

    value_set.assign(l1_lhs, l1_rhs, ns);
  }
}

void goto_symex_statet::rename(expr2tc &expr)
{
  // rename all the symbols with their last known value

  if (is_nil_expr(expr))
    return;

  if (is_symbol2t(expr))
  {
    top().level1.rename(expr);
    level2.rename(expr);
  }
  else if (is_address_of2t(expr))
  {
    address_of2t &addrof = to_address_of2t(expr);
    rename_address(addrof.ptr_obj);
  }
  else
  {
    // do this recursively
    std::vector<expr2tc*> operands;
    expr.get()->list_operands(operands);
    for (std::vector<expr2tc*>::iterator it = operands.begin();
         it != operands.end(); it++)
      rename(**it);
  }
}

void goto_symex_statet::rename_address(expr2tc &expr)
{
  // rename all the symbols with their last known value

  if (is_symbol2t(expr))
  {
    // only do L1
    top().level1.rename(expr);
  }
  else if (is_index2t(expr))
  {
    index2t &index = to_index2t(expr);
    rename_address(index.source_value);
    rename(index.index);
  }
  else
  {
    // do this recursively
    std::vector<expr2tc*> operands;
    expr.get()->list_operands(operands);
    for (std::vector<expr2tc*>::iterator it = operands.begin();
         it != operands.end(); it++)
      rename_address(**it);
  }
}

void goto_symex_statet::get_original_name(expr2tc &expr) const
{
  std::vector<expr2tc*> operands;
  expr.get()->list_operands(operands);
  for (std::vector<expr2tc*>::iterator it = operands.begin();
       it != operands.end(); it++)
    get_original_name(**it);

  if (is_symbol2t(expr))
  {
    level2.get_original_name(expr);
    top().level1.get_original_name(expr);
  }
}

const irep_idt goto_symex_statet::get_original_name(
  const irep_idt &identifier) const
{

  return top().level1.get_original_name(
         level2.get_original_name(identifier));
}

void goto_symex_statet::print_stack_trace(unsigned int indent) const
{
  call_stackt::const_reverse_iterator it;
  symex_targett::sourcet src;
  std::string spaces = std::string("");
  unsigned int i;

  for (i = 0; i < indent; i++)
    spaces += " ";

  // Iterate through each call frame printing func name and location.
  src = source;
  for (it = call_stack.rbegin(); it != call_stack.rend(); it++) {
    if (it->function_identifier == "") { // Top level call
      std::cout << spaces << "init" << std::endl;
    } else {
      std::cout << spaces << it->function_identifier.as_string();
      std::cout << " at " << src.pc->location.get_file();
      std::cout << " line " << src.pc->location.get_line();
      std::cout << std::endl << std::endl;
    }

    src = it->calling_location;
  }

  if (!thread_ended) {
    std::cout << spaces << "Next instruction to be executed:" << std::endl;
    source.prog->output_instruction(ns, "", std::cout, source.pc, true, false);
  }

  return;
}

std::vector<dstring>
goto_symex_statet::gen_stack_trace(void) const
{
  std::vector<dstring> trace;
  call_stackt::const_reverse_iterator it;
  symex_targett::sourcet src;
  int i = 0;

  // Format is a vector of strings, each recording a particular function
  // invocation.

  for (it = call_stack.rbegin(); it != call_stack.rend(); it++) {
    src = it->calling_location;

    if (it->function_identifier == "") { // Top level call
      break;
    } else if (it->function_identifier == "c::main" &&
               src.pc->location == get_nil_irep()) {
      trace.push_back("<main invocation>");
    } else {
      std::string loc = it->function_identifier.as_string();
      loc += " at " + src.pc->location.get_file().as_string();
      loc += " line " + src.pc->location.get_line().as_string();
      trace.push_back(loc);
    }
  }

  return trace;
}
